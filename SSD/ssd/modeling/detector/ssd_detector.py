from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections

    def forward_without_postprocess(self, images, softmax, targets=None, ):
        features = self.backbone(images)
        cls_scores, reg_scores, cls_scores_presoftmax = self.box_head.forward_without_postprocess(
            features, softmax, targets)

        return (cls_scores, reg_scores, features, cls_scores_presoftmax)

    def get_boxes(self, cls_logits, bbox_pred):
        detections, _ = self.box_head._forward_test_without_softmax(
            cls_logits, bbox_pred)
        return detections

    def get_features(self, images, targets=None):
        #features = self.backbone(images)

        features = self.backbone.get_feats(images)

        # print(features.shape)
        # exit()

        return features

    def forward_ood(self, images, targets):
        features = self.backbone(images)
        detections, detector_losses = self.box_head._forward_ood(
            features, targets)
        if self.training:
            return detector_losses
        return features, detections, detector_losses
