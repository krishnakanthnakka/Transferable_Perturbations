import torch
import numpy as np
import torchvision
import os
import glob
import random
import logging
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt


from PIL import Image
from .base_model import BaseModel
from . import networks
from PIL import Image
from utils import do_detect, plot_boxes
from cda.data.transforms import build_transforms
from pix2pix.models.resnet_gen import GeneratorResnet, weights_init_normal
from vizer.draw import draw_boxes
from torchvision.utils import save_image
from cda.engine.inference import do_evaluation, do_evaluation_adv
from advertorch.utils import clamp
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from cda.modeling.detector import build_detection_model, build_detection_model_eval
from cda.modeling.detector.at import AT
from cda.modeling.detector.loss import attention_loss, feat_loss_mutliscale_fn, gram_loss_multiscale_fn
from .visualize_filters import FilterVisualizer
from cv2 import resize


def normalize_fn(t):
    t = t.clone()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


def de_normalize_fn(t):
    t = t.clone()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t


def subtract_mean(img, mean):

    out = img.clone()
    out[:, 0] = img[:, 0] - mean[0]
    out[:, 1] = img[:, 1] - mean[1]
    out[:, 2] = img[:, 2] - mean[2]
    return out


class ClassifierModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=500, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        logger = logging.getLogger("CDA.inference")

        self.num_classes = opt.detcfg.MODEL.NUM_CLASSES
        self.loss_names = ['ce']
        self.visual_names = ['search_clean_vis', 'search_adv_vis']
        if self.isTrain:
            self.model_names = ['C']
        else:  # during test time, only load G
            self.model_names = ['C']

        self.iter = 0
        self.scale = [0, 1]
        self.attackobjective = opt.attackobjective
        self.cpu_device = torch.device("cpu")
        self.perturbmode = opt.perturbmode

        if self.attackobjective == 'Blind':
            self.netC = build_detection_model(
                opt.detcfg, opt.act_layer, opt.act_layer_mean, pretrained=False).cuda()
            self.netC = torch.nn.DataParallel(self.netC).cuda()

            if 0 and self.isTrain:
                logger.info("Applying Random Normal Initialization!")
                self.netC.apply(weights_init_normal)

            self.netC.train()

            # self.optimizer_G = torch.optim.Adam(self.netC.parameters(), lr=opt.lr,
            #                                     betas=(opt.beta1, 0.999))

            print(self.netC.parameters())

            self.optimizer_G = torch.optim.SGD(self.netC.parameters(), opt.lr,
                                               momentum=0.9,
                                               weight_decay=1e-4)

            self.optimizers.append(self.optimizer_G)

        self.det_mean = opt.detcfg.INPUT.PIXEL_MEAN

    def load_eval_model(self):

        if not self.opt.detcfg.EVAL_MODEL.EVAL_DIFFERENT_MODEL_FROM_TRAIN:
            self.eval_classifier = self.netC
        else:
            self.eval_classifier = build_detection_model_eval(
                self.opt.detcfg, self.opt.act_layer, self.opt.act_layer_mean).cuda()

    def set_input(self, input):

        self.clean1 = input[0].cuda()
        self.targets = input[1].cuda()

        assert torch.max(self.clean1) <= 1, 'Wrong Normalization with max value :{}'.format(
            torch.max(self.clean1))
        assert torch.min(self.clean1) >= 0, 'Wrong Normalization'

        self.image_ids = input[2]

    def forward():

        return

    def optimize_parameters(self):

        self.netC.train()
        loss_dict = {}
        criterion = torch.nn.CrossEntropyLoss()

        self.logits_clean = self.netC(normalize_fn(self.clean1.clone()))
        loss_ce = criterion(self.logits_clean, self.targets)
        loss_dict['ce'] = loss_ce

        self.optimizer_G.zero_grad()
        loss_ce.backward()
        self.optimizer_G.step()

        save_image(self.clean1, 'clean.png', nrow=4)

        return loss_dict

    def evaluate(self, iteration, save_feats):

        # self.load_eval_model()
        self.netC.eval()
        eval_results, self.clean_predictions_eval = do_evaluation(self.opt.detcfg,
                                                                  self.netC,
                                                                  distributed=False,
                                                                  save_feats=save_feats,
                                                                  num_images=self.opt.num_images,
                                                                  train_config=self.opt.train_config,
                                                                  eval_model=self.opt.eval_model)

        return eval_results[0]['metrics']
