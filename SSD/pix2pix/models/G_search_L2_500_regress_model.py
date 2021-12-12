import torch
import logging
from .base_model import BaseModel
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
from ssd.data.datasets import VOCDataset
from ssd.engine.inference import do_evaluation, do_evaluation_adv
from pix2pix.models.resnet import GeneratorResnet
class_names = VOCDataset.class_names


def normalize_to_minus_1_to_plus_1(im_tensor):
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor


def subtract_mean(img, mean):

    out = img.clone()

    out[:, 0] = img[:, 0] - mean[0]
    out[:, 1] = img[:, 1] - mean[1]
    out[:, 2] = img[:, 2] - mean[2]
    return out


def add_mean(img, mean):

    out = img.clone()
    out[:, 0] = img[:, 0] + mean[0]
    out[:, 1] = img[:, 1] + mean[1]
    out[:, 2] = img[:, 2] + mean[2]
    return out


def load_ssd_model(opt):
    model = build_detection_model(opt.detcfg)
    checkpointer = CheckPointer(model, save_dir=opt.detcfg.OUTPUT_DIR)
    model = model.cuda().eval()
    checkpointer.load(opt.detector_ckpt, strict=False)
    return model


class GsearchL2500regressModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='unet_256',
                            dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float,
                                default=500, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.num_classes = opt.detcfg.MODEL.NUM_CLASSES
        self.loss_names = ['G_L2', 'cls', 'reg', 'conf', 'feat']
        self.visual_names = ['search_clean_vis', 'search_adv_vis']
        self.model_names = ['G']

        self.netG = GeneratorResnet().cuda()

        self.attackobjective = opt.attackobjective
        self.cpu_device = torch.device("cpu")
        self.perturbmode = opt.perturbmode
        self.logger = logging.getLogger("SSD.training")

        self.siam = load_ssd_model(opt)
        self.logger.info("Loaded Detector!!")

        self.det_mean = opt.detcfg.INPUT.PIXEL_MEAN
        self.eps = opt.eps
        self.opt.detckptname = self.opt.detector_ckpt.split("/")[-2]

    def set_input(self, input):

        self.clean255 = input[0].cuda()
        self.targets = input[1]

        assert torch.max(self.clean255) <= 255, 'Wrong Normalization'
        assert torch.min(self.clean255) >= 0, 'Wrong Normalization'

        self.clean1 = normalize_to_minus_1_to_plus_1(self.clean255)
        self.clean255_det_normed = subtract_mean(self.clean255, self.det_mean)
        self.image_ids = input[2]

    def forward(self, target_sz=(300, 300)):
        image512_clean1 = torch.nn.functional.interpolate(self.clean1,
                                                          size=(512, 512), mode='bilinear')

        if self.perturbmode:
            perturb = self.netG(image512_clean1)
            image512_adv1 = image512_clean1 + perturb
        else:
            # NIPS approach
            image512_adv1 = self.netG(image512_clean1)

        self.adv1 = torch.nn.functional.interpolate(image512_adv1, size=target_sz, mode='bilinear')
        self.adv1 = torch.min(torch.max(self.adv1, self.clean1 - self.eps), self.clean1 + self.eps)
        self.adv1 = torch.clamp(self.adv1, -1.0, 1.0)

        # self.adv1 = self.clean1

        self.adv255 = self.adv1 * 127.5 + 127.5
        self.adv255_det_normed = subtract_mean(self.adv255, self.det_mean)

        return self.adv255_det_normed

    def backward_G(self):
        pass

    def optimize_parameters(self):
        pass

    def evaluate(self, iteration, save_feats):
        do_evaluation(self.opt.detcfg, self.siam, distributed=False,
                      iteration=iteration,
                      save_feats=save_feats,
                      mean=self.det_mean,
                      det_name=self.opt.detckptname,
                      pooling_type=self.opt.pooling_type,
                      num_images=self.opt.num_images)

    def evaluate_adv(self, iteration, save_feats, gen_size):

        do_evaluation_adv(self.opt.detcfg,
                          self.siam,
                          distributed=False,
                          generator=self.netG,
                          eps=self.eps,
                          mean=self.det_mean,
                          iteration=iteration,
                          save_feats=save_feats,
                          det_name=self.opt.detckptname,
                          pooling_type=self.opt.pooling_type,
                          num_images=self.opt.num_images,
                          perturbmode=self.perturbmode,
                          gen_size=gen_size)
