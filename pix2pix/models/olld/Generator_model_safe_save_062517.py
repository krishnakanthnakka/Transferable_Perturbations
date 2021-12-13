import torch
import numpy as np
import torchvision
import os
import glob
import random

from .base_model import BaseModel
from . import networks
from PIL import Image
from utils import do_detect, plot_boxes
import torch.nn.functional as F
from cda.data.transforms import build_transforms
from pix2pix.models.resnet import GeneratorResnet
# from cda.engine.inference_utils import preprocess, entropy_loss, generate_mask
# Detector imports
# from ssd.modeling.detector import build_detection_model
# from ssd.utils.checkpoint import CheckPointer
from vizer.draw import draw_boxes
from torchvision.utils import save_image
# , do_evaluation_adv, do_evaluation_ood, get_mask_from_rois, lossfn_feat
from cda.engine.inference import do_evaluation
from advertorch.utils import clamp
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
# class_names = VOCDataset.class_names
from cda.modeling.detector import build_detection_model
from cda.modeling.detector.at import AT
from cda.modeling.detector.loss import attention_loss, feat_loss


def normalize_fn(t):

    t = t.clone()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def normalize_to_minus_1_to_plus_1(im_tensor):
    '''(0,255) ---> (-1,1)'''
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


def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)

    return x


def L2_dist(x, y):
    return reduce_sum((x - y) ** 2)


def load_ssd_model(opt):
    model = build_detection_model(opt.detcfg)
    checkpointer = CheckPointer(
        model, save_dir=opt.detcfg.OUTPUT_DIR)

    # CHANGED
    model = model.cuda().eval()
    # model = model.cuda().train()

    checkpointer.load(opt.detckpt)

    return model


class GeneratorModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='unet_256',
                            dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float,
                                default=500, help='weight for L1 loss')

        return parser

    def load_gan(self, opt):
        #  print('Label: {} \t Attack: {} dependent \t Model: {} \t Distribution: {} \t Saving instance: {}'.format(args.target,
        #                                                                                                          args.attack_type,
        #                                                                                                          args.model_type,
        #                                                                                                          args.train_dir,
        #                                                                                                          args.epochs))
        # model_name = 'netG_{}_{}_{}.pth'.format(
        #     args.target, args.attack_type, args.epochs)

        # OUTPUT_DIR = 'saved_models/{}/{}_{}_{}'.format(
        #     args.model_type, args.train_dir, args.loss_type, args.run)

        # full_path = os.path.join(OUTPUT_DIR, model_name)

        full_path = "./netG_-1_img_9.pth"
        print("Loading model from :{}".format(full_path))

        self.netG.load_state_dict(torch.load(full_path))

        return

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

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
            self.netG = GeneratorResnet().cuda()

        if self.opt.pretrained_netG:
            self.load_gan(opt)

            print(self.netG.parameters())

        self.scale = [0, 1]

        self.attackobjective = opt.attackobjective
        self.cpu_device = torch.device("cpu")

        self.perturbmode = opt.perturbmode

        # print(self.perturbmode)
        # exit()

        if self.isTrain:

            self.netG.train()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionLfeat = torch.nn.MSELoss(reduction='sum')
            self.weight_L2 = opt.weight_L2
            self.weight_ce = opt.weight_ce
            self.weight_rl = opt.weight_rl
            self.weight_att = opt.weight_att
            self.weight_feat = opt.weight_feat
            self.cls_margin = opt.cls_margin

            opt.log("Weights, L2:{}, ce:{}, rl;{}, feat:{} att:{}".format(
                self.weight_L2, self.weight_ce, self.weight_rl, self.weight_feat, self.weight_att))

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if self.attackobjective == 'Blind':
            self.classifier = build_detection_model(opt.detcfg).cuda()
            self.det_mean = opt.detcfg.INPUT.PIXEL_MEAN

            self.eps = opt.eps

        # self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        #                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        # self.target_feature()
        self.opt.detckptname = self.opt.detckpt.split("/")[-2]

    def set_input(self, input):

        self.clean1 = input[0].cuda()
        self.targets = input[1]

        assert torch.max(self.clean1) <= 1, 'Wrong Normalization with max value :{}'.format(
            torch.max(self.clean1))
        assert torch.min(self.clean1) >= 0, 'Wrong Normalization'

        if 0 and random.uniform(0, 1) > 1.5:
            self.clean255_OOD = self.generate_OODs(self.clean255).cuda()
            self.clean255 = torch.cat((self.clean255, self.clean255_OOD), 0)

        # self.clean255_det_normed = subtract_mean(self.clean255, self.det_mean)
        self.image_ids = input[2]
        # exit()

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

        # image512_clean1 = torch.nn.functional.interpolate(self.clean1,
        #                                                   size=(512, 512), mode='bilinear')
        # CHANGED MAJOR

        if self.perturbmode:
            perturb = self.netG(self.clean1)
            self.adv1 = self.clean1 + perturb
        else:
            # --------  NIPS approach
            self.adv1 = self.netG(self.clean1)

        self.adv1 = torch.min(torch.max(self.adv1, self.clean1 - self.eps),
                              self.clean1 + self.eps)

        self.adv1 = torch.clamp(self.adv1, 0.0, 1.0)

        return self.adv1

    def get_mask_from_rois(self):

        N, C, H, W = self.clean255.shape
        mask_ = np.zeros((N, 1, H, W), dtype=np.float32)

        for idx in range(N):
            # print(idx)
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
            loss_rl = -criterion(self.adv_logits -
                                 self.clean_logits, self.pred_label) * weights['rl']
            loss_dict['rl'] = loss_rl
            self.loss_rl = loss_rl

        if 'ce' in self.opt.loss_type:
            loss_ce = -criterion(self.adv_logits,
                                 self.pred_label) * self.weight_ce
            loss_dict['ce'] = loss_ce
            self.loss_ce = loss_ce

        if 'feat' in self.opt.loss_type:
            self.criterionfeat = torch.nn.MSELoss()
            loss_feat = -feat_loss(self.feats_adv, self.feats_clean,
                                   self.criterionfeat) * self.weight_feat
            loss_dict['feat'] = loss_feat
            self.loss_feat = loss_feat

        if 'att' in self.opt.loss_type:
            self.criterionAT = AT(att_order)
            loss_att = - \
                attention_loss(self.feats_clean, self.feats_adv,
                               self.criterionAT) * weights['att']
            loss_dict['att'] = loss_att
            self.loss_att = loss_att

        self.loss_G = self.loss_G_L2 + self.loss_ce + \
            self.loss_rl + self.loss_feat + self.loss_att + self.loss_ce
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

        self.forward()
        self.logits_adv, self.feats_adv = self.classifier(
            normalize_fn(self.adv1), return_feats=True)

        self.optimizer_G.zero_grad()
        loss_dict = self.backward_G()
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

        # ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
        #            cmap=matplotlib.colors.ListedColormap(colors), marker='<', s=40)

        for j in range(len(np.unique(labels))):
            ax.scatter(X_2d[labels == j, 0], X_2d[labels == j, 1], c=labels[labels == j],
                       cmap=matplotlib.colors.ListedColormap(colors[j]), marker=markers[j], s=40)

        fig.savefig('./misc/training/layer_{}.png'.format(layer),
                    dpi=150, bbox_inches='tight')

        return X_2d

    def save_clean_and_adv(self):
        results_dict = {}
        cpu_device = torch.device("cpu")
        score_threshold = 0.3

        with torch.no_grad():
            outputs = self.classifier(normalize_fn(self.clean1))
            outputs = [o.to(cpu_device) for o in outputs]
            results_dict.update(
                {int(img_id): result for img_id,
                 result in zip(self.image_ids, outputs)}
            )

        clean_images = self.clean255.clone()

        for b in range(clean_images.shape[0]):
            image = (self.clean255.data[b].cpu().
                     numpy()).transpose(1, 2, 0)
            image = image.astype(np.uint8)

            class_names = self.opt.dataset.class_names

            result = outputs[b]
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']
            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            drawn_image = draw_boxes(image, boxes, labels,
                                     scores, class_names).astype(np.uint8)
            drawn_image = drawn_image.transpose(2, 0, 1)
            clean_images[b] = torch.tensor(drawn_image)

            Image.fromarray(image).save(os.path.join(
                '/cvlabdata1/home/krishna/DA/SSD_Attacks', './results/clean_{}.png'.format(b)))

        save_image(clean_images / 255, os.path.join(
            '/cvlabdata1/home/krishna/DA/SSD_Attacks', 'clean.png'), nrow=4)

        with torch.no_grad():
            outputs = self.siam(self.adv255_det_normed)
            outputs = [o.to(cpu_device) for o in outputs]
            results_dict.update(
                {int(img_id): result for img_id,
                 result in zip(self.image_ids, outputs)}
            )

        adv_images = self.clean255.clone()

        for b in range(adv_images.shape[0]):

            image = (self.adv255.data[b].cpu().
                     numpy()).transpose(1, 2, 0)
            # print(np.max(image), np.min(image))
            image = image.astype(np.uint8)
            class_names = self.opt.dataset.class_names
            result = outputs[b]
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']
            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            drawn_image = draw_boxes(image, boxes, labels,
                                     scores, class_names).astype(np.uint8)
            drawn_image = drawn_image.transpose(2, 0, 1)
            # print(drawn_image.shape, np.max(drawn_image), np.min(drawn_image))
            adv_images[b] = torch.tensor(drawn_image)
            Image.fromarray(image).save(os.path.join(
                '/cvlabdata1/home/krishna/DA/SSD_Attacks', './results/adv_{}.png'.format(b)))

        save_image(adv_images / 255, os.path.join(
            '/cvlabdata1/home/krishna/DA/SSD_Attacks', 'adv.png'), nrow=4)

    def evaluate(self, iteration, save_feats):
        eval_results = do_evaluation(
            self.opt.detcfg, self.classifier, distributed=False, iteration=iteration, save_feats=save_feats, mean=self.det_mean, det_name=self.opt.detckptname, pooling_type=self.opt.pooling_type, num_images=self.opt.num_images)

    def evaluate_adv(self, iteration, save_feats):
        eval_results = do_evaluation_adv(
            self.opt.detcfg, self.siam, distributed=False, generator=self.netG, eps=self.eps, mean=self.det_mean,
            iteration=iteration, save_feats=save_feats, det_name=self.opt.detckptname, pooling_type=self.opt.pooling_type, num_images=self.opt.num_images, perturbmode=self.perturbmode)

    def evaluate_OOD(self, iteration, save_feats):
        eval_results = do_evaluation_ood(
            self.opt.detcfg, self.siam, distributed=False, generator=self.netG, eps=self.eps, mean=self.det_mean,
            iteration=iteration, save_feats=save_feats, det_name=self.opt.detckptname, pooling_type=self.opt.pooling_type, num_images=self.opt.num_images, perturbmode=self.perturbmode)
