from cda.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
#from .transforms import *
import torchvision.transforms as transforms


def build_transforms(cfg, is_train=True, data_aug_transforms=False):

    #print("YO", is_train)

    # if is_train:
    #     transform = [
    #         ConvertFromInts(),
    #         PhotometricDistort(),
    #         Expand(cfg.INPUT.PIXEL_MEAN),
    #         RandomSampleCrop(),
    #         RandomMirror(),
    #         ToPercentCoords(),
    #         Resize(cfg.INPUT.IMAGE_SIZE),
    #         SubtractMeans(cfg.INPUT.PIXEL_MEAN),
    #         ToTensor(),
    #     ]
    # else:
    #     transform = [
    #         Resize(cfg.INPUT.RESIZE_SIZE),
    #         CenterCrop(cfg.INPUT.IMAGE_SIZE),
    #         ToTensor(),
    #     ]
    # transform = Compose(transform)

    if is_train:
        # transform = transforms.Compose(
        #     [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), ])
        #
        if data_aug_transforms:
            transform = transforms.Compose(
                [transforms.Resize(256),
                 # transforms.CenterCrop(224),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ToTensor(),
                 ])  #

        else:

            # FOR TRAINING THE PERTURBATION GENERATOR
            transform = transforms.Compose(
                [
                    transforms.Resize(cfg.INPUT.RESIZE_SIZE),
                    transforms.CenterCrop(cfg.INPUT.IMAGE_SIZE),
                    # transforms.Resize(224),  # Added this
                    # transforms.RandomCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ])  #

        import logging
        logger = logging.getLogger("CDA.trainer")
        logger.info("Training Transforms: {}".format(transform))

    else:
        transform = transforms.Compose([
            transforms.Resize(cfg.INPUT.RESIZE_SIZE),
            transforms.CenterCrop(cfg.INPUT.IMAGE_SIZE),
            transforms.ToTensor(), ])

        # # CHANGED
        # transform = transforms.Compose([
        #     transforms.Resize(300),
        #     transforms.CenterCrop(299),
        #     transforms.ToTensor(), ])
        import logging
        logger = logging.getLogger("CDA.trainer")
        logger.info("Testing Transforms: {}".format(transform))

    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
