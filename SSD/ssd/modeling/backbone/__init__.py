from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .mobilenetv3 import MobileNetV3
from .resnet50 import resnet50_SSD300

__all__ = ['build_backbone', 'VGG', 'MobileNetV2',
           'EfficientNet', 'MobileNetV3', 'resnet50_SSD300']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
