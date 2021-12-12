from torch import nn
import torch.nn.functional as F

from ssd.modeling import registry
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.box_head.box_predictor import make_box_predictor
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(
            neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)

        #print(cls_logits.shape, bbox_pred.shape)
        # exit()

        if self.training:

            # CHANGED MAJOR
            # return self._forward_train(cls_logits, bbox_pred, targets)
            return self._forward_test(cls_logits, bbox_pred)

        else:
            return self._forward_test(cls_logits, bbox_pred)

    def forward_without_postprocess(self, features, softmax, targets=None):
        cls_logits_presoftmax, bbox_pred = self.predictor(features)

        if softmax:
            cls_logits = F.softmax(cls_logits_presoftmax, dim=2)
            return (cls_logits, bbox_pred, cls_logits_presoftmax)
        else:
            return (cls_logits_presoftmax, bbox_pred, cls_logits_presoftmax)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)

        # CHANGED AND REMOVED
        # bbox_pred[:, :2] = 0.0
        # bbox_pred[:, 2:] = 1.0

        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)

        detections = self.post_processor(detections)
        return detections, {}

    def _forward_test_without_softmax(self, scores, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        #scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)

        detections = self.post_processor(detections)
        return detections, {}

    # This function is used exclusively in OOD generation process
    def _forward_ood(self, features, targets):
        cls_logits_presoftmax, bbox_pred = self.predictor(features)

        gt_boxes, gt_labels = targets['boxes'].cuda(), targets['labels'].cuda()
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits_presoftmax, bbox_pred, gt_labels, gt_boxes)

        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )

        # --------------------- CHANGED:  PLEASE CHECK THE CODE. UNINTENDED BEHAVIOUR MAY BE THERE -----------------------
        # for only positives boxes or all boxes
        cls_logits_presoftmax_pos = cls_logits_presoftmax[gt_labels >= 0]
        # CHANGED to only pos for logits
        detections = (cls_logits_presoftmax_pos, bbox_pred)

        scores = F.softmax(cls_logits_presoftmax, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        loss_dict['outputs'] = self.post_processor((scores, boxes))
        return detections, loss_dict
