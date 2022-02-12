import torch
import os
import random
import logging

from .base_model import BaseModel
from pix2pix.models.resnet_gen import GeneratorResnet, weights_init_normal
from torchvision.utils import save_image
from cda.engine.inference import do_evaluation, do_evaluation_adv
from cda.modeling.detector import build_detection_model, build_detection_model_eval
from cda.modeling.detector.at import AT
from cda.modeling.detector.loss import attention_loss, feat_loss_mutliscale_fn


def normalize_fn(t):
    t = t.clone()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


class GeneratorModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=500, help='weight for L1 loss')
        return parser

    def load_gan(self, opt):

        logger = logging.getLogger("CDA.inference")
        full_path = self.opt.pretrain_weights
        logger.info("Loading model from :{}".format(full_path))
        checkpoint = torch.load(full_path)

        self.netG.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint.keys():
            self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

        # new additions
        state = torch.load(os.path.join(os.path.dirname(full_path), 'optimizer.pth'))
        self.optimizer_G.load_state_dict(state['state_dict'])
        logger.info("Loading optimizer from :{}".format(
            os.path.join(os.path.dirname(full_path), 'optimizer.pth')))

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        logger = logging.getLogger("CDA.inference")
        self.num_classes = opt.detcfg.MODEL.NUM_CLASSES
        self.loss_names = ['G_L2', 'ce', 'rl', 'att', 'feat']
        self.visual_names = ['search_clean_vis', 'search_adv_vis']
        self.model_names = ['G']

        if opt.train_getG_299:
            self.netG = GeneratorResnet(opt.gen_dropout, opt.data_dim,
                                        inception=True, isTrain=self.isTrain).cuda()
            logger.info("Generator model with Inception flag")

        else:
            self.netG = GeneratorResnet(opt.gen_dropout, opt.data_dim, isTrain=self.isTrain).cuda()
            logger.info("Generator model Without Inception flag")

        self.loss_fn, self.SOFTMAX_2D = feat_loss_mutliscale_fn, opt.softmax2D

        if self.isTrain:
            logger.info("Loss Type: {},  SOFTMAX_2D: {}".format(
                self.opt.loss_type, self.SOFTMAX_2D))

            for i, (name, layer) in enumerate(self.netG.block1.named_parameters()):
                logger.info("{}, {}, {}, {}, {}, {}".format(i, name, layer.shape,
                                                            torch.mean(layer), torch.max(layer), torch.min(layer)))

        self.iter = 0
        self.scale = [0, 1]
        self.cpu_device = torch.device("cpu")
        self.perturbmode = opt.perturbmode

        if self.isTrain:
            self.netG.train()
            self.critrionL2 = torch.nn.L1Loss()
            self.criterionLfeat = torch.nn.MSELoss(reduction='sum')
            self.weight_L2 = opt.weight_L2
            self.weight_ce = opt.weight_ce
            self.weight_rl = opt.weight_rl
            self.weight_att = opt.weight_att
            self.weight_feat = opt.weight_feat
            self.cls_margin = opt.cls_margin
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.classifier = build_detection_model(
                opt.detcfg, opt.act_layer, opt.act_layer_mean).cuda()
            self.classifier.eval()

        self.det_mean = opt.detcfg.INPUT.PIXEL_MEAN
        self.eps = opt.eps
        self.opt.detckptname = "vgg16"

        if self.opt.pretrained_netG:
            self.load_gan(opt)

        self.netG = torch.nn.DataParallel(self.netG).cuda()

        if self.isTrain:
            self.criterionfeat = torch.nn.MSELoss()
            logger.info("Loss fn: {}".format(self.criterionfeat))

    def load_eval_model(self):

        logger = logging.getLogger("CDA.inference")

        if not self.opt.detcfg.EVAL_MODEL.EVAL_DIFFERENT_MODEL_FROM_TRAIN:
            self.eval_classifier = self.classifier
            logger.info(" Evaluating using saame model at train time!!")

        else:
            self.eval_classifier = build_detection_model_eval(
                self.opt.detcfg, self.opt.act_layer, self.opt.act_layer_mean).cuda()

    def set_input(self, input):

        self.clean1 = input[0].cuda()
        self.targets = input[1]

        assert torch.max(self.clean1) <= 1, 'Wrong Normalization with max value :{}'.format(
            torch.max(self.clean1))
        assert torch.min(self.clean1) >= 0, 'Wrong Normalization'

        self.image_ids = input[2]

    def forward(self, target_sz=(300, 300)):

        assert torch.max(self.clean1) <= 1, 'Wrong Normalization with max value :{}'.format(
            torch.max(self.clean1))
        assert torch.min(self.clean1) >= 0, 'Wrong Normalization'

        if self.perturbmode:
            perturb = self.netG(self.clean1)
            self.adv1 = self.clean1 + perturb
        else:
            # --------  NIPS approach
            self.adv1 = self.netG(self.clean1)

            if False and self.adv1.shape[-1] != self.clean1.shape[-1]:
                self.adv1 = torch.nn.functional.interpolate(self.adv1, size=self.clean1.shape[2:],
                                                            mode='bilinear')
            self.adv1_unbounded = self.adv1.clone()

        self.adv1 = torch.min(torch.max(self.adv1, self.clean1 - self.eps), self.clean1 + self.eps)
        self.adv1 = torch.clamp(self.adv1, 0.0, 1.0)
        self.perturb = torch.abs(self.adv1 - self.clean1)

        return self.adv1

    def backward_G(self):
        self.loss_feat = torch.tensor(0.0)
        self.loss_G_L2 = torch.tensor(0.0)
        self.loss_ce = torch.tensor(0.0)
        self.loss_rl = torch.tensor(0.0)
        self.loss_att = torch.tensor(0.0)

        if self.weight_L2 > 1e-6:
            self.criterionL2 = torch.nn.MSELoss()
            self.loss_G_L2 = self.criterionL2(self.adv1,
                                              self.clean1) * self.weight_L2

        loss = 0.0
        loss_dict = {}

        if 'feat' in self.opt.loss_type:

            if self.weight_feat > 0:
                loss_feat = -self.loss_fn(self.feats_adv, self.feats_clean,
                                          self.criterionfeat, SOFTMAX_2D=self.SOFTMAX_2D) * self.weight_feat
                loss_dict['feat'] = loss_feat
                self.loss_feat = loss_feat

        self.loss_G = self.loss_G_L2 + self.loss_ce + \
            self.loss_rl + self.loss_feat + self.loss_att

        self.loss_G.backward(retain_graph=True)

        lossdict = {}
        lossdict['L2'] = self.loss_G_L2
        lossdict['ce'] = self.loss_ce
        lossdict['rl'] = self.loss_rl
        lossdict['att'] = self.loss_att
        lossdict['feat'] = self.loss_feat
        return lossdict

    def optimize_parameters(self):

        with torch.no_grad():
            self.logits_clean, self.feats_clean = self.classifier(
                normalize_fn(self.clean1.detach()), return_feats=True)

        self.preds_clean = torch.argmax(self.logits_clean, dim=-1).detach()
        self.netG.train()
        self.forward()
        self.logits_adv, self.feats_adv = self.classifier(normalize_fn(self.adv1),
                                                          return_feats=True)

        self.optimizer_G.zero_grad()
        loss_dict = self.backward_G()

        self.optimizer_G.step()
        return loss_dict

    def update_writer(self, iteration, grad=True):

        for tag, parm in self.netG.named_parameters():
            self.writer.add_histogram(tag, parm.data.cpu().numpy(), iteration)
            if grad:
                self.writer.add_histogram(tag + "_grad", parm.grad.data.cpu().numpy(), iteration)

    def save_clean_and_adv(self, epoch):
        results_dict = {}
        cpu_device = torch.device("cpu")
        score_threshold = 0.3

        with torch.no_grad():
            outputs = self.classifier(normalize_fn(self.clean1))
            outputs = [o.to(cpu_device) for o in outputs]
        clean_images = self.clean1.clone()
        save_image(clean_images, os.path.join(self.opt.detcfg.OUTPUT_DIR, 'clean.png'), nrow=4)

        with torch.no_grad():
            outputs = self.classifier(normalize_fn(self.adv1))
            outputs = [o.to(cpu_device) for o in outputs]

        save_image(self.adv1, os.path.join(self.opt.detcfg.OUTPUT_DIR,
                                           'adv_{}.png'.format(epoch)), nrow=4)

        save_image(self.adv1_unbounded, os.path.join(self.opt.detcfg.OUTPUT_DIR,
                                                     'adv_unbounded_{}.png'.format(epoch)), nrow=4)

        save_image(5 * self.perturb + 0.5, os.path.join(self.opt.detcfg.OUTPUT_DIR,
                                                        'perturb_{}.png'.format(epoch)), nrow=4)

        if not os.path.exists(os.path.join(self.opt.detcfg.OUTPUT_DIR, 'images')):
            os.makedirs(os.path.join(self.opt.detcfg.OUTPUT_DIR, 'images'))

        self.iter += 1

    def evaluate(self, iteration, save_feats):

        self.load_eval_model()
        eval_results, self.clean_predictions_eval = do_evaluation(self.opt.detcfg,
                                                                  self.eval_classifier,
                                                                  distributed=False,
                                                                  save_feats=save_feats,
                                                                  num_images=self.opt.num_images,
                                                                  train_config=self.opt.train_config,
                                                                  eval_model=self.opt.eval_model,
                                                                  save_layer_index=self.opt.act_layer[0])

        # print(self.clean_predictions_eval[0]))

    def evaluate_adv(self, iteration, save_feats):

        self.load_eval_model()
        self.netG.eval()
        logger = logging.getLogger("CDA.inference")

        self.eval_classifier.eval()

        eval_results, self.adv_predictions_eval = do_evaluation_adv(self.opt.detcfg,
                                                                    self.eval_classifier,
                                                                    distributed=False,
                                                                    save_feats=save_feats,
                                                                    num_images=self.opt.num_images,
                                                                    generator=self.netG,
                                                                    eps=self.eps,
                                                                    train_config=self.opt.train_config,
                                                                    eval_model=self.opt.eval_model,
                                                                    perturbmode=self.perturbmode,
                                                                    save_layer_index=self.opt.act_layer[0])

        self.netG.train()
        return eval_results[0]['metrics']
