from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True, is_train_OOD=False):

    # print(cfg.INPUT.PIXEL_MEAN)

    if is_train:

        print("Safe")
        if not is_train_OOD:
            print("Not Using OOD data transforms")
            # exit()
            transform = [

                # ConvertFromInts(),
                # PhotometricDistort(),   # THIS PULLS OUTSIDE 0 to 255 range
                # Expand(cfg.INPUT.PIXEL_MEAN),
                # RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                # SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]

            """ CHANGED
                ConvertFromInts(),
                # PhotometricDistort(),   # THIS PULLS OUTSIDE 0 to 255 range
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                # SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
                """

        else:
            print("Using OOD data transforms")
            transform = [
                # ConvertFromInts(),
                # PhotometricDistort(),
                # Expand(cfg.INPUT.PIXEL_MEAN),
                # RandomSampleCrop(),
                # RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                # SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor()
            ]

    else:
        transform = [
            # ConvertFromInts(),
            # PhotometricDistort(),
            # Expand(cfg.INPUT.PIXEL_MEAN),
            # RandomSampleCrop(),
            # RandomMirror(),
            # ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
