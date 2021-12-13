import torch
import numpy as np
import torchvision
import os
import glob
import random
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


def normalize_fn(t):
    t = t.clone()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


def subtract_mean(img, mean):

    out = img.clone()
    out[:, 0] = img[:, 0] - mean[0]
    out[:, 1] = img[:, 1] - mean[1]
    out[:, 2] = img[:, 2] - mean[2]
    return out


def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def L2_dist(x, y):
    return reduce_sum((x - y) ** 2)


def load_ssd_model(opt):
    model = build_detection_model(opt.detcfg)
    checkpointer = CheckPointer(model, save_dir=opt.detcfg.OUTPUT_DIR)
    model = model.cuda().eval()
    checkpointer.load(opt.detckpt)
    return model


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
        full_path = os.path.join(self.opt.pretrain_weights)
        logger.info("Loading model from :{}".format(full_path))

        # self.netG.load_state_dict(torch.load(full_path))

        checkpoint = torch.load(full_path)
        self.netG.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        logger = logging.getLogger("CDA.inference")

        self.num_classes = opt.detcfg.MODEL.NUM_CLASSES
        self.loss_names = ['G_L2', 'ce', 'rl', 'att', 'feat']
        self.visual_names = ['search_clean_vis', 'search_adv_vis']
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        # self.netG = networks.define_G(opt.input_nc, 3, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.train_classifier == 'incv3':
            self.netG = GeneratorResnet(inception=True).cuda()
        else:
            self.netG = GeneratorResnet(opt.gen_dropout, opt.data_dim).cuda()
            # self.netG = self.netG.train()

        if False:
            logger.info(" Applying Random Normal Initialization!")
            self.netG.apply(weights_init_normal)

        # To be Removed
        # self.loss_fn, self.SOFTMAX_2D = loss_mutliscale, False  # gram_loss_fn, True

        self.loss_fn, self.SOFTMAX_2D = opt.loss_fn, opt.softmax2D  # gram_loss_fn, True

        logger.info("Loss Type: {}, Loss Fn: {}, SOFTMAX_2D: {}".format(
            self.opt.loss_type, self.loss_fn.__name__, self.SOFTMAX_2D))

        for i, (name, layer) in enumerate(self.netG.block1.named_parameters()):
            logger.info("{}, {}, {}, {}, {}, {}".format(i, name, layer.shape,
                                                        torch.mean(layer), torch.max(layer), torch.min(layer)))

        self.iter = 0
        self.scale = [0, 1]
        self.attackobjective = opt.attackobjective
        self.cpu_device = torch.device("cpu")
        self.perturbmode = opt.perturbmode

        if self.isTrain:
            self.netG.train()
            self.criterionL2 = torch.nn.L1Loss()
            self.criterionLfeat = torch.nn.MSELoss(reduction='sum')
            self.weight_L2 = opt.weight_L2
            self.weight_ce = opt.weight_ce
            self.weight_rl = opt.weight_rl
            self.weight_att = opt.weight_att
            self.weight_feat = opt.weight_feat
            self.cls_margin = opt.cls_margin

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))  # opt.bera1

            # self.optimizer_G = torch.optim.SGD(            #     self.netG.parameters(), lr=opt.lr, momentum=0.9)  # opt.bera1

            self.optimizers.append(self.optimizer_G)

            if self.attackobjective == 'Blind':
                self.classifier = build_detection_model(
                    opt.detcfg, opt.act_layer, opt.act_layer_mean).cuda()
                self.classifier.eval()

        self.det_mean = opt.detcfg.INPUT.PIXEL_MEAN
        self.eps = opt.eps
        self.opt.detckptname = "vgg16"

        if self.opt.pretrained_netG:
            self.load_gan(opt)

        self.netG = torch.nn.DataParallel(self.netG).cuda()

    def load_eval_model(self):

        if not self.opt.detcfg.EVAL_MODEL.EVAL_DIFFERENT_MODEL_FROM_TRAIN:
            self.eval_classifier = self.classifier
        else:
            self.eval_classifier = build_detection_model_eval(
                self.opt.detcfg, self.opt.act_layer, self.opt.act_layer_mean).cuda()

    def set_input(self, input):

        self.clean1 = input[0].cuda()
        self.targets = input[1]

        assert torch.max(self.clean1) <= 1, 'Wrong Normalization with max value :{}'.format(
            torch.max(self.clean1))
        assert torch.min(self.clean1) >= 0, 'Wrong Normalization'

        if 0 and random.uniform(0, 1) > 1.5:
            self.clean255_OOD = self.generate_OODs(self.clean255).cuda()
            self.clean255 = torch.cat((self.clean255, self.clean255_OOD), 0)

        self.image_ids = input[2]

    def generate_OODs(self, images):

        idx = 0
        loops_adv = 50  # CHANGED
        weight_eta = 1.0
        gamma = 1.0
        lr = 0.5
        weight_decay = 0.005
        momentum = 0.9
        CLIP = True
        eps_pgd = 30 / 255
        eps_iter = 3 / 255  # CHANGED
        images = images / 255.0
        cpu_device = torch.device("cpu")

        score_maps_clean_raw, reg_maps_clean, feats_clean, score_maps_clean_raw_presoftmax = self.siam.forward_without_postprocess(
            preprocess(images.detach().clone(), self.det_mean), softmax=True)

        clean_detections = self.siam.get_boxes(
            score_maps_clean_raw, reg_maps_clean)
        clean_detections = [o.to(cpu_device)
                            for o in clean_detections]

        att_masks = get_mask_from_rois(
            images * 255, clean_detections).cuda()

        inputs_max = images.detach().clone().cuda()

        delta = torch.zeros_like(inputs_max).uniform_(-eps_pgd, eps_pgd)
        delta.requires_grad_()

        for ite_max in range(loops_adv):

            perturbed_inputs = inputs_max + 1 * delta
            perturbed_inputs = torch.clamp(
                perturbed_inputs, 0.0, 1.0)

            preprocessed_inputs = preprocess(perturbed_inputs, self.det_mean)

            features_ood, cls_box_logits, loss_dict = self.siam.forward_ood(
                preprocessed_inputs, self.targets)

            outputs = [o.to(cpu_device) for o in loss_dict['outputs']]

            ood_images = inputs_max.clone()

            if ite_max == loops_adv - 1:
                for k in range(images.shape[0]):
                    result = outputs[k]
                    boxes, labels, scores = result['boxes'], result['labels'], result['scores']
                    indices = scores > 0.5
                    boxes, labels, scores = boxes[indices], labels[indices], scores[indices]
                    mask = generate_mask(boxes, images)

                    image_cpu = (((inputs_max[k] + delta[k]) * 255).detach().cpu().
                                 numpy()).transpose(1, 2, 0).astype(np.uint8)

                    drawn_image = draw_boxes(image_cpu, boxes.detach(), labels,
                                             scores, class_names).astype(np.uint8)
                    # Image.fromarray(drawn_image).save(os.path.join(
                    #     '/cvlabdata1/home/krishna/DA/SSD_Attacks', './results/adv_{}.png'.format(k)))
                    # Image.fromarray(image_cpu).save(os.path.join(
                    #    '/cvlabdata1/home/krishna/DA/SSD_Attacks', './results/images_{}.png'.format(k)))
                    ood_images[k] = torch.tensor(
                        drawn_image.transpose(2, 0, 1))

                save_image(
                    ood_images / 255, os.path.join('/cvlabdata1/home/krishna/DA/SSD_Attacks/results/', 'ood_{}.png'.format(idx)), nrow=4)
                # exit()

            loss_cls = loss_dict['cls_loss']
            loss_entropy = weight_eta * entropy_loss(cls_box_logits[0])

            loss_feat = -gamma * \
                (lossfn_feat(att_masks, feats_clean, features_ood))
            # CHANGED  from feats_clean

            loss = 0 * loss_cls + 0 * loss_entropy - 1 * loss_feat

            if ite_max < loops_adv - 1:
                # model.zero_grad()
                # optimizer.zero_grad()
                # (-loss).backward(retain_graph=True)
                # optimizer.step()

                loss.backward(retain_graph=True)
                grad_sign = delta.grad.data.sign()
                delta.data = delta.data + \
                    batch_multiply(eps_iter, grad_sign)
                delta.data = batch_clamp(eps_pgd, delta.data)
                delta.data = clamp(inputs_max.data + delta.data, 0.0, 1.0
                                   ) - inputs_max.data

                delta.grad.data.zero_()

            # inputs_max = torch.clamp(inputs_max, 0, 1.0)

            if ite_max % 10 == -1:
                print('ite_adv: {:3d}, Loss: {:.5f}, Loss_cls:{:.5f}, Loss_entropy: {:.5f}, Loss_feat: {:.5f} '.format(
                    ite_max, loss.item(), loss_cls.item(), loss_entropy.item(), loss_feat.item()))

        inputs_max = perturbed_inputs
        inputs_max = inputs_max.detach().clone().cpu()

        return inputs_max * 255.0

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

    def get_mask_from_rois(self):

        N, C, H, W = self.clean255.shape
        mask_ = np.zeros((N, 1, H, W), dtype=np.float32)

        for idx in range(N):
            image = (self.clean255.data[idx].cpu().
                     numpy()).transpose(1, 2, 0).astype(np.uint8)
            output = self.clean_detections[idx]
            rois, scores = output['boxes'], output['scores']
            # print(rois, scores)
            mask = np.zeros((H, W), dtype=np.float32)
            rois_num = len(scores)
            ymins = []
            xmins = []
            ymaxs = []
            xmaxs = []
            for i in range(rois_num):
                ymins.append(max(0, int(rois[i][1])))
                xmins.append(max(0, int(rois[i][0])))
                ymaxs.append(min(H, int(rois[i][3])))
                xmaxs.append(min(W, int(rois[i][2])))
            for i in range(rois_num):
                h = ymaxs[i] - ymins[i]
                w = xmaxs[i] - xmins[i]
                if h < 0 or w < 0:
                    continue
                roi_weight = np.ones((h, w)) * scores[i].item()
                if h == 0 or w == 0:
                    mask = mask + 0
                else:
                    mask[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]] = mask[ymins[i]:ymaxs[i],
                                                                      xmins[i]:xmaxs[i]] + roi_weight
            mask_min = np.min(mask)
            mask_max = np.max(mask)

            if mask_max - mask_min > 0:
                mask = (mask - mask_min) / (mask_max - mask_min)
            mask_[idx, 0] = mask
        #     Image.fromarray(image).save(os.path.join(
        #         '/cvlabdata1/home/krishna/DA/SSD_Attacks', './results/clean_{}.png'.format(idx)))
        #     Image.fromarray((mask * 255).astype('uint8')).save(os.path.join(
        #         '/cvlabdata1/home/krishna/DA/SSD_Attacks', './results/cleanmask_{}.png'.format(idx)))
        # exit()
        return torch.tensor(mask_)

    def lossfn_feat(self):

        att_masks = self.get_mask_from_rois().cuda()
        loss = torch.tensor(0.0).cuda()

        for j in range(1, 2):
            h, w = self.feats_clean[j].shape[2:]
            att_masks_ = F.interpolate(att_masks, (h, w))
            # feat_target = self.feats_target
            feat_target = self.feats_clean

            # print(feat_target[j].shape, att_masks_.shape,
            #       self.feats_clean[j].shape)
            # print(att_masks_[0])
            # exit()

            # REMOVED BELOW

            # CHANGED
            if False:
                loss = loss + self.criterionLfeat(self.feats_adv[j] * att_masks_,
                                                  feat_target[j] * att_masks_)

            else:

                # Version 1
                # f1 = F.normalize(feat_target[j], p=2, dim=1)
                # f2 = F.normalize(self.feats_adv[j], p=2, dim=1)
                # loss = loss + self.criterionLfeat(f1, f2)

                # Version 2
                # f1 = torch.mean(feat_target[j], (2, 3))
                # f2 = torch.mean(self.feats_adv[j], (2, 3))
                # f1 = F.normalize(f1, p=2, dim=-1)
                # f2 = F.normalize(f2, p=2, dim=-1)
                # loss = loss + self.criterionLfeat(f1, f2)

                # VERSION 3
                loss = loss + self.criterionLfeat(self.feats_adv[j],
                                                  feat_target[j])

        # exit()

        return -loss

    def target_feature(self):

        image_paths = glob.glob(os.path.join(
            '/cvlabdata1/home/krishna/DA/SSD_Attacks', '*.jpg'))

        cpu_device = torch.device("cpu")
        transforms = build_transforms(self.opt.detcfg, is_train=False)
        for i, image_path in enumerate(image_paths):
            image_name = os.path.basename(image_path)
            image = np.array(Image.open(image_path).convert("RGB"))
            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0).cuda()
            score_maps_target_raw, reg_maps_target, self.feats_target, _ = self.siam.forward_without_postprocess(images,
                                                                                                                 softmax=True)
            target_detections = self.siam.get_boxes(score_maps_target_raw,
                                                    reg_maps_target)
            target_detections = [o.to(self.cpu_device)
                                 for o in target_detections]
        return

    def backward_G(self):
        self.loss_feat = torch.tensor(0.0)
        self.loss_G_L2 = torch.tensor(0.0)
        self.loss_ce = torch.tensor(0.0)
        self.loss_rl = torch.tensor(0.0)
        self.loss_att = torch.tensor(0.0)
        if self.opt.weight_L2 > 0:
            self.loss_G_L2 = self.criterionL2(self.adv1,
                                              self.clean1) * self.weight_L2

        loss = 0.0
        loss_dict = {}

        if 'rl' in self.opt.loss_type:
            criterion = torch.nn.CrossEntropyLoss()
            loss_rl = -criterion(self.logits_adv -
                                 self.logits_clean, self.preds_clean) * self.weight_rl
            loss_dict['rl'] = loss_rl
            self.loss_rl = loss_rl

        if 'ce' in self.opt.loss_type:
            criterion = torch.nn.CrossEntropyLoss()
            loss_ce = -criterion(self.logits_adv,
                                 self.preds_clean) * self.weight_ce
            loss_dict['ce'] = loss_ce
            self.loss_ce = loss_ce

        if 'feat' in self.opt.loss_type:

            if self.weight_feat > 0:
                self.criterionfeat = torch.nn.MSELoss()

                # loss_fn, SOFTMAX_2D = gram_loss_fn, True
                # loss_fn = feat_loss
                # loss_fn = loss_mutliscale
                # SOFTMAX_2D = True

                loss_feat = -self.loss_fn(self.feats_adv, self.feats_clean,
                                          self.criterionfeat, SOFTMAX_2D=self.SOFTMAX_2D) * self.weight_feat
                loss_dict['feat'] = loss_feat
                self.loss_feat = loss_feat

        if 'att' in self.opt.loss_type:
            self.criterionAT = AT(self.opt.att_order)
            loss_att = - \
                attention_loss(self.feats_clean, self.feats_adv,
                               self.criterionAT) * self.weight_att
            loss_dict['att'] = loss_att
            self.loss_att = loss_att

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

        # CHANGED
        # torch.nn.utils.clip_grad_norm(self.netG.parameters(), 1.0)

        self.optimizer_G.step()
        return loss_dict

    def plot_tSNE(self, feats, labels, num_classes, layer, label_names):

        print(np.unique(labels))

        cm = plt.get_cmap('Paired')
        colors = [cm(1. * i / num_classes) for i in range(num_classes)]
        # label_names = [str(i) for i in range(num_classes)]
        # tsne = TSNE(n_components=2, random_state=0, n_iter=500, verbose=1,
        #             n_iter_without_progress=800, learning_rate=20, perplexity=15)
        tsne = TSNE(perplexity=15, n_iter=5000, n_jobs=10,
                    n_iter_without_progress=800, learning_rate=20, metric='cosine')

        print(" TSNE Running!")
        X_2d = tsne.fit_transform(feats)
        print(" TSNE Done!")
        markers = [8, 9, '<', '>', '<', 9]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex='col',
                               sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

        wrong_h_single_ind = [plt.plot([], [], color=colors[i], marker=markers[i], ls="", markersize=10)[
            0] for i in range(len(colors))]
        wrong_legend_adv = ax.legend(handles=wrong_h_single_ind, labels=label_names, loc=(
            1.02, 0), framealpha=0.3, prop={'size': 13})
        wrong_legend_adv.set_title("Labels", prop={'size': 13})
        ax.add_artist(wrong_legend_adv)

        for j in range(len(np.unique(labels))):
            ax.scatter(X_2d[labels == j, 0], X_2d[labels == j, 1], c=labels[labels == j],
                       cmap=matplotlib.colors.ListedColormap(colors[j]), marker=markers[j], s=40)

        fig.savefig('./misc/training/layer_{}.png'.format(layer),
                    dpi=150, bbox_inches='tight')

        return X_2d

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
        save_image(self.adv1, os.path.join(self.opt.detcfg.OUTPUT_DIR, 'images',
                                           'adv_{}.png'.format(self.iter)), nrow=4)

        self.iter += 1

    def evaluate(self, iteration, save_feats):

        self.load_eval_model()
        eval_results, self.clean_predictions_eval = do_evaluation(self.opt.detcfg,
                                                                  self.eval_classifier,
                                                                  distributed=False,
                                                                  save_feats=save_feats,
                                                                  num_images=self.opt.num_images,
                                                                  train_config=self.opt.train_config,
                                                                  eval_model=self.opt.eval_model)

        # print(self.clean_predictions_eval[0]))

    def evaluate_adv(self, iteration, save_feats):

        self.load_eval_model()
        self.netG.eval()

        eval_results, self.adv_predictions_eval = do_evaluation_adv(self.opt.detcfg,
                                                                    self.eval_classifier,
                                                                    distributed=False,
                                                                    save_feats=save_feats,
                                                                    num_images=self.opt.num_images,
                                                                    generator=self.netG,
                                                                    eps=self.eps,
                                                                    train_config=self.opt.train_config,
                                                                    eval_model=self.opt.eval_model,
                                                                    perturbmode=self.perturbmode)

        self.netG.train()

    def pgd(self):

        import pickle

        delta = torch.zeros_like(self.clean1).uniform_(-self.eps, self.eps)
        delta.data = torch.max(torch.min(1 - self.clean1, delta.data), 0 - self.clean1)

        self.criterionfeat = torch.nn.MSELoss()

        iters = 200
        alpha = 2 / 255.0

        layer_name = 'feat_3'

        features_normal_dict = {}
        features_adv_dict = {}

        self.logits_clean, self.feats_clean = self.classifier(
            normalize_fn(self.clean1.detach()), return_feats=True)

        preds_clean = self.logits_clean.argmax(dim=-1)
        acc_clean = torch.sum(preds_clean == self.targets.cuda()) / \
            self.clean1.shape[0]
        print("acc_clean: {}".format(acc_clean))

        # f = torch.mean(self.feats_clean['fc'], (2, 3))
        # np.savetxt("clean.txt", f.cpu().detach().numpy(), '%4d')

        features_normal = []
        features_normal.append(self.feats_clean[layer_name].detach().cpu().numpy())
        features_normal = np.concatenate(features_normal)

        features_normal_dict['feats'] = features_normal

        with open("./normal.pkl", 'wb') as f:
            pickle.dump(features_normal_dict, f, pickle.HIGHEST_PROTOCOL)

        # print(preds_clean, self.targets)

        for i in range(iters):
            delta.requires_grad = True

            self.logits_adv, self.feats_adv = self.classifier(
                normalize_fn((self.clean1 + delta)), return_feats=True)

            # loss = self.criterionfeat(
            #     self.feats_clean['fc'], self.feats_adv['fc'][:, 4])

            # loss = self.loss_fn(torch.mean(self.feats_adv, (2, 3)), torch.mean(self.feats_clean, (2, 3)),
            #                     self.criterionfeat, SOFTMAX_2D=self.SOFTMAX_2D) * self.weight_feat

            loss = self.loss_fn(self.feats_adv, self.feats_clean,
                                self.criterionfeat, SOFTMAX_2D=self.SOFTMAX_2D) * self.weight_feat

            # f = torch.mean(self.feats_adv['fc'], (2, 3))
            # f = torch.amax(self.feats_adv['fc'], dim=(2, 3))

            # B, C = self.feats_adv['fc'].shape[:2]

            # f = self.feats_adv['fc'].view(B * C, -1)

            # np.savetxt("./misc/adv_{}.txt".format(i), f.cpu().detach().numpy(), '%4d')

            # loss = torch.mean(torch.clamp(
            #     -self.feats_adv['fc'] + self.feats_clean['fc'], min=-25))

            # margin = 25
            # feat_fc = self.feats_clean['fc'].view(self.feats_clean['fc'].shape[0], -1)
            # feat_adv = self.feats_adv['fc'].view(self.feats_adv['fc'].shape[0], -1)
            # loss = torch.mean((feat_fc - feat_adv)**2 * (torch.abs(feat_fc - feat_adv) < margin))

            # loss = -self.feats_adv['fc'].std()

            loss.backward(retain_graph=True)
            print(i, loss.item())
            grad = delta.grad.detach()
            delta.data = torch.clamp(delta + alpha * torch.sign(grad), -self.eps, self.eps)
            delta.data = torch.max(torch.min(1 - self.clean1, delta.data), 0 - self.clean1)
            delta.grad.data.zero_()

        delta = delta.detach()
        adv = torch.clamp(self.clean1 + delta, 0.0, 1.0)
        logits_adv, feats_adv = self.classifier(normalize_fn(adv.detach()), return_feats=True)

        # f = torch.mean(feats_adv['fc'], (2, 3))
        # np.savetxt("adv.txt", f.cpu().detach().numpy(), '%4d')

        preds_adv = logits_adv.argmax(dim=-1)
        accuracy = torch.sum(preds_clean != preds_adv) / self.clean1.shape[0]

        save_image(self.clean1, "clean1.png")
        save_image(adv, "adv1.png")
        print("Fooling rate:{}".format(accuracy))

        features_adv = []
        features_dict_adv = {}
        features_adv.append(feats_adv[layer_name].detach().cpu().numpy())
        features_adv = np.concatenate(features_adv)

        features_dict_adv['feats'] = features_adv

        print(features_adv.shape)

        with open("./adv.pkl", 'wb') as f:
            pickle.dump(features_dict_adv, f, pickle.HIGHEST_PROTOCOL)

        exit()

        return adv.detach()

    def savefeatfigs(self, feats):

        plt.figure(figsize=(30, 30))
        feat = feats['feat_3'].squeeze().detach().cpu().numpy()
        for i, filter in enumerate(feat):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.savefig("./misc/feat.png")

        exit()

    def cross_domain_attack(self):

        imgpaths = '/cvlabdata1/home/krishna/DA/CDA/DUP/CDA/cda/data/datasets/imagenet5k_val.txt'
        files = open(imgpaths).readlines()
        criterion = torch.nn.MSELoss()
        import torchvision.transforms as transforms
        import PIL
        layer_name = 'feat_10'
        eps = 10.0 / 255.0
        alpha = 2 / 255.0

        transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             ])  #

        saveimg = torch.zeros((5, 3, 224, 224)).cuda()

        preds_clean = []
        gt = []
        preds_adv = []

        # self.eval_classifier = build_detection_model_eval(self.opt.detcfg).cuda().eval()
        self.eval_classifier = self.classifier

        for j in range(16):
            # print(files[j])
            L = files[j + self.iter * 16].rstrip()
            L_ = L.split(" ")
            gt.append(int(L_[-1]))
            imgname = ' '.join(L_[:-1])
            image = Image.open(imgname).convert("RGB")
            image = np.array(image)
            image = Image.fromarray(image)
            image = transform(image).unsqueeze(0).cuda()

            logits_clean, feats_clean_target = self.classifier(
                normalize_fn(image.detach()), return_feats=True)
            preds_clean_imagenet = logits_clean.argmax(dim=-1)
            preds_clean.append(int(preds_clean_imagenet[0].cpu().numpy()))

            saveimg[0] = image

            image_chestx = self.clean1[j].view(1, 3, 224, 224)
            # image_chestx = self.clean1[j].view(1, 3, 224, 224) * 0 + 0.5

            saveimg[1] = image_chestx

            _, feats_clean_chestx = self.classifier(
                normalize_fn(image_chestx.detach()), return_feats=True)

            self.savefeatfigs(feats_clean_target)

            # print(feats_clean_chestx[layer_name].shape)

            delta = torch.zeros_like(image_chestx)  # .uniform_(-eps, eps)
            delta.requires_grad_()

            for it in range(1500):

                _, features_ood = self.classifier(normalize_fn(
                    (image_chestx + delta).clone()), True)

                # loss = -1 * \
                #     criterion(features_ood[layer_name], feats_clean_target[layer_name])

                # print(features_ood['feat_3'].shape)

                loss = -1 * \
                    feat_loss_mutliscale_fn(features_ood, feats_clean_target,
                                            criterion, SOFTMAX_2D=True)

                # loss = -1 * loss_gram_mutliscale(features_ood, feats_clean_target, criterion)

                loss.backward(retain_graph=True)
                grad = delta.grad
                delta.data = torch.clamp(delta.data + alpha * grad.sign(), -eps, eps)
                delta.data = clamp(image_chestx.data + delta.data, 0.0, 1.0) - image_chestx
                delta.grad.data.zero_()

                if it % 50 == 0:
                    print('iter: {:3d}, Loss: {:.6f}'.format(it, loss.item()))

            delta = delta.detach()
            ood = torch.clamp(image_chestx + 1 * delta, 0.0, 1.0)
            # print(255 * torch.max(torch.abs(ood - image_chestx)))

            saveimg[2] = ood

            # ---  RUN PGD attack on perturbed chestx

            _, feats_clean_ood = self.eval_classifier(
                normalize_fn(ood.detach()), return_feats=True)
            alpha2 = 2 / 255.0
            eps2 = 10 / 255.0
            delta_ood = torch.zeros_like(ood).uniform_(-eps2, eps2)
            delta_ood.data = torch.max(torch.min(1 - ood, delta_ood.data), 0 - ood)

            # ll = 'feat_3'

            for i in range(51):
                delta_ood.requires_grad = True
                _, feats_adv_ood = self.eval_classifier(
                    normalize_fn((ood + delta_ood)), return_feats=True)

                # loss = criterion(feats_clean_ood[ll], feats_adv_ood[ll])

                loss = feat_loss_mutliscale_fn(
                    feats_clean_ood, feats_adv_ood, criterion, SOFTMAX_2D=True)

                loss.backward(retain_graph=True)
                if i % 10 == 0:
                    print('iter: {:3d}, Loss: {:.6f}'.format(i, loss.item()))

                grad = delta_ood.grad.detach()
                delta_ood.data = torch.clamp(
                    delta_ood + alpha2 * torch.sign(grad), -eps2, eps2)
                delta_ood.data = torch.max(torch.min(1 - ood, delta_ood.data), 0 - ood)
                delta_ood.grad.data.zero_()

            delta_ood = delta_ood.detach()

            saveimg[3] = delta_ood + ood
            saveimg[4] = delta_ood + image

            adv_imagenet = torch.clamp(image + delta_ood, 0.0, 1.0)
            logits_adv, feats_adv = self.classifier(
                normalize_fn(adv_imagenet.detach()), return_feats=True)
            preds_adv_imagenet = logits_adv.argmax(dim=-1)
            preds_adv.append(int(preds_adv_imagenet[0].cpu().numpy()))

            print("{:3d}, {:3d}({:.2f}), {:3d}({:.2f}), {:.1f}\n".format(j, preds_clean_imagenet[0]
                                                                         .item(), F.softmax(logits_clean, dim=-1).max(dim=-1)[0].item(), preds_adv_imagenet[0]
                                                                         .item(), F.softmax(logits_adv, -1).max(dim=-1)[0].item(),
                                                                         torch.max(torch.abs(adv_imagenet - image) * 255)))

            save_image(saveimg, "misc/{}.png".format(j))

        gt, preds_clean, preds_adv = np.array(gt), np.array(preds_clean), np.array(preds_adv)

        print("GT:{}, Normal:{}, Adv:{}".format(gt, preds_clean, preds_adv))
        clean_acc = np.mean(gt == preds_clean)
        fooling_rate = np.mean(preds_clean != preds_adv)

        print(clean_acc, fooling_rate, )
        self.iter += 1

    def cross_domain_attack_optim(self):

        imgpaths = '/cvlabdata1/home/krishna/DA/CDA/DUP/CDA/cda/data/datasets/imagenet5k_val.txt'
        files = open(imgpaths).readlines()
        criterion = torch.nn.MSELoss()
        import torchvision.transforms as transforms
        import PIL
        layer_name = 'feat_10'
        eps = 10.0 / 255.0
        alpha = 2 / 255.0

        transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             ])  #

        saveimg = torch.zeros((5, 3, 224, 224)).cuda()

        preds_clean = []
        gt = []
        preds_adv = []

        # self.eval_classifier = build_detection_model_eval(self.opt.detcfg).cuda().eval()
        self.eval_classifier = self.classifier

        for j in range(16):
            L = files[j].rstrip()
            L_ = L.split(" ")
            gt.append(int(L_[-1]))
            imgname = ' '.join(L_[:-1])
            image = Image.open(imgname).convert("RGB")
            image = np.array(image)
            image = Image.fromarray(image)
            image = transform(image).unsqueeze(0).cuda()

            logits_clean, feats_clean_target = self.classifier(
                normalize_fn(image.detach()), return_feats=True)
            preds_clean_imagenet = logits_clean.argmax(dim=-1)
            preds_clean.append(int(preds_clean_imagenet[0].cpu().numpy()))

            saveimg[0] = image

            image_chestx = self.clean1[j].view(1, 3, 224, 224)

            saveimg[1] = image_chestx

            _, feats_clean_chestx = self.classifier(
                normalize_fn(image_chestx.detach()), return_feats=True)

            target = image_chestx.clone()

            # target = normalize_fn(target).clone()
            # target = target.requires_grad_(True)

            # optimizer = torch.optim.Adam([target])  # opt.bera1
            # for it in range(1000):
            #     target.data.clamp_(0, 1)
            #     optimizer.zero_grad()

            #     # _, features_ood = self.classifier(target, True)
            #     _, features_ood = self.classifier(normalize_fn(target.clone()), True)

            #     # loss = -1 * \
            #     #     criterion(features_ood[layer_name], feats_clean_target[layer_name])

            #     loss = feat_loss_mutliscale_fn(
            #         features_ood, feats_clean_target, criterion, SOFTMAX_2D=True)
            #     loss.backward(retain_graph=True)
            #     optimizer.step()

            #     if it % 50 == 0:
            #         print('iter: {:3d}, Loss: {:.20f}'.format(it, loss.item()))

            print('Optimizing..')
            target = target.requires_grad_(True)
            optimizer = torch.optim.LBFGS([target])
            run = [0]
            while run[0] <= 1000:
                def closure():
                    # target.data.clamp_(0, 1)
                    optimizer.zero_grad()
                    _, features_ood = self.classifier(normalize_fn(target.clone()), True)

                    # loss = criterion(features_ood[layer_name], feats_clean_target[layer_name])

                    loss = feat_loss_mutliscale_fn(
                        features_ood, feats_clean_target, criterion, SOFTMAX_2D=False)

                    loss.backward(retain_graph=True)

                    if run[0] % 50 == 0:
                        print('iter: {:4d}, Loss: {:.20f}'.format(run[0], loss.item()))

                    run[0] += 1
                    return loss

                optimizer.step(closure)

            ood = torch.clamp(target, 0.0, 1.0)
            print(255 * torch.max(torch.abs(ood - image_chestx)))

            saveimg[2] = ood

            # ---  RUN PGD attack on perturbed chestx

            _, feats_clean_ood = self.eval_classifier(
                normalize_fn(ood.detach()), return_feats=True)
            alpha2 = 2 / 255.0
            eps2 = 10 / 255.0
            delta_ood = torch.zeros_like(ood).uniform_(-eps2, eps2)
            delta_ood.data = torch.max(torch.min(1 - ood, delta_ood.data), 0 - ood)

            ll = 'feat_3'

            for i in range(51):
                delta_ood.requires_grad = True
                _, feats_adv_ood = self.eval_classifier(
                    normalize_fn((ood + delta_ood)), return_feats=True)

                loss = criterion(feats_clean_ood[ll], feats_adv_ood[ll])

                # loss = loss_mutliscale(feats_clean_ood, feats_adv_ood, criterion)

                loss.backward(retain_graph=True)
                if i % 10 == 0:
                    print(i, loss.item())
                grad = delta_ood.grad.detach()
                delta_ood.data = torch.clamp(
                    delta_ood + alpha2 * torch.sign(grad), -eps2, eps2)
                delta_ood.data = torch.max(torch.min(1 - ood, delta_ood.data), 0 - ood)
                delta_ood.grad.data.zero_()

            delta_ood = delta_ood.detach()

            saveimg[3] = delta_ood + ood
            saveimg[4] = delta_ood + image

            adv_imagenet = torch.clamp(image + delta_ood, 0.0, 1.0)
            logits_adv, feats_adv = self.classifier(
                normalize_fn(adv_imagenet.detach()), return_feats=True)
            preds_adv_imagenet = logits_adv.argmax(dim=-1)
            preds_adv.append(int(preds_adv_imagenet[0].cpu().numpy()))

            print("{:3d}, {:3d}({:.2f}), {:3d}({:.2f}), {:.1f}\n".format(j,
                                                                         preds_clean_imagenet[0]
                                                                         .item(),
                                                                         F.softmax(logits_clean,
                                                                                   dim=-1).max(dim=-1)[0].item(),
                                                                         preds_adv_imagenet[0]
                                                                         .item(),
                                                                         F.softmax(
                                                                             logits_adv, -1).max(dim=-1)[0]
                                                                         .item(),
                                                                         torch.max(torch.abs(adv_imagenet - image) * 255)))

            save_image(saveimg, "misc/{}.png".format(j))

            # if j == 2:
            #     break

        gt, preds_clean, preds_adv = np.array(gt), np.array(preds_clean), np.array(preds_adv)

        print(gt, preds_clean, preds_adv)
        clean_acc = np.mean(gt == preds_clean)
        fooling_rate = np.mean(preds_clean != preds_adv)

        print(clean_acc, fooling_rate, )


# ---------------------------------------------------------- OLD ------------------------------------------------------------------------------------
    # def evaluate_OOD(self, iteration, save_feats):
    #     eval_results = do_evaluation_ood(
    #         self.opt.detcfg, self.siam, distributed=False, generator=self.netG, eps=self.eps, mean=self.det_mean,
    #         iteration=iteration, save_feats=save_feats, det_name=self.opt.detckptname, pooling_type=self.opt.pooling_type, num_images=self.opt.num_images, perturbmode=self.perturbmode)
# define the VGG
# class StyleNet(nn.Module):
#     import torch.nn as nn
#     import torchvision.models as models
#     import torch
#     def __init__(self):
#         super(StyleNet, self).__init__()
#         self.StyleNet = models.vgg19(pretrained=True).features
#     def get_style_activations(self, x):
#         """
#             Extracts the features for the style loss from the block1_conv1,
#                 block2_conv1, block3_conv1, block4_conv1, block5_conv1 of VGG19
#             Args:
#                 x: torch.Tensor - input image we want to extract the features of
#             Returns:
#                 features: list - the list of activation maps of the block1_conv1,
#                     block2_conv1, block3_conv1, block4_conv1, block5_conv1 layers
#         """
#         features = [self.StyleNet[:4](x)] + [self.StyleNet[:7](x)] + [self.StyleNet[:12](x)
#                                                                       ] + [self.StyleNet[:21](x)] + [self.StyleNet[:30](x)]
#         return features
#     def forward(self, x):
#         return self.StyleNet(x)
