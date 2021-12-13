from .vgg import vgg16, vgg19
from .resnet import resnet152
from .squeezenet import squeezenet1_1
from .densenet import densenet121, densenet201, densenet161, densenet169
from .inception import inception_v3
import torch
from .mobilnet import mobilenet_v3_large
from .mnasnet import mnasnet1_0
from .shufflenet import shufflenet_v2_x1_0
#from .inception_v4 import inception_v4


_DETECTION_META_ARCHITECTURES = {
    "vgg16": vgg.vgg16,
    "vgg11": vgg.vgg11,
    "vgg13": vgg.vgg13,
    "vgg19": vgg.vgg19,
    'resnet152': resnet.resnet152,
    'squeezenet1_1': squeezenet1_1,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'inception_v3': inception_v3,
    'resnet50': resnet.resnet50,
    'resnet18': resnet.resnet18,
    'resnet101': resnet.resnet101,
    'resnet34': resnet.resnet34,
    'mobilenet_v3': mobilnet.mobilenet_v3_large,
    'mnasnet1_0': mnasnet.mnasnet1_0,
    'shufflenetv2': shufflenet.shufflenet_v2_x1_0,
    # 'inception_v4': inception_v4,

}


def build_detection_model(cfg, act_layer, act_layer_mean, pretrained=True):

    if cfg.MODEL.META_ARCHITECTURE == 'chestxNet':
        from .DensenetModels import DenseNet121
        model = DenseNet121(14, True, act_layer=act_layer, act_layer_mean=act_layer_mean).cuda()
        model_multi = torch.nn.DataParallel(model).cuda()
        modelCheckpoint = torch.load(
            '/cvlabdata1/home/krishna/DA/Cross-domain-perturbations/pretrained/m-25012018-123527.pth.tar')
        state_dict = modelCheckpoint['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            str_ = key.replace("norm.0", "norm0")
            str_ = str_.replace("norm.1", "norm1")
            str_ = str_.replace("norm.2", "norm2")
            str_ = str_.replace("conv.0", "conv0")
            str_ = str_.replace("conv.1", "conv1")
            str_ = str_.replace("conv.2", "conv2")
            new_state_dict[str_] = state_dict[key]

        model_multi.load_state_dict(new_state_dict)
        # print("Loaded pretrained ChestxNet classifier")
        import logging
        logger = logging.getLogger("CDA.inference")
        logger.info("Loaded pretrained ChestxNet classifier for training phase")

        return model

    else:
        meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
        # print(cfg)

        # exit()
        # print("\n\nYo")
        # print(cfg)
        return meta_arch(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, num_classes=cfg.MODEL.NUM_CLASSES, act_layer=act_layer, act_layer_mean=act_layer_mean)


def build_detection_model_eval(cfg, act_layer, act_layer_mean, pretrained=True):

    if cfg.EVAL_MODEL.META_ARCHITECTURE == 'chestxNet':
        from .DensenetModels import DenseNet121
        model = DenseNet121(14, True, act_layer=act_layer, act_layer_mean=act_layer_mean).cuda()
        model_multi = torch.nn.DataParallel(model).cuda()
        modelCheckpoint = torch.load(
            '/cvlabdata1/home/krishna/DA/Cross-domain-perturbations/pretrained/m-25012018-123527.pth.tar')
        state_dict = modelCheckpoint['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            str_ = key.replace("norm.0", "norm0")
            str_ = str_.replace("norm.1", "norm1")
            str_ = str_.replace("norm.2", "norm2")
            str_ = str_.replace("conv.0", "conv0")
            str_ = str_.replace("conv.1", "conv1")
            str_ = str_.replace("conv.2", "conv2")
            new_state_dict[str_] = state_dict[key]

        model_multi.load_state_dict(new_state_dict)
        # print("Loaded pretrained ChestxNet classifier")
        import logging
        logger = logging.getLogger("CDA.inference")
        logger.info(
            "Loaded pretrained ChestxNet classifier for evaluaation phase")

        return model

    else:
        meta_arch = _DETECTION_META_ARCHITECTURES[cfg.EVAL_MODEL.META_ARCHITECTURE]
        return meta_arch(pretrained=cfg.EVAL_MODEL.BACKBONE.PRETRAINED, num_classes=cfg.EVAL_MODEL.NUM_CLASSES, act_layer=act_layer, act_layer_mean=act_layer_mean)
