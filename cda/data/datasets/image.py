import os
import torch.utils.data
import numpy as np
from PIL import Image
import logging
dir_path = os.path.dirname(os.path.realpath(__file__))


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, images_path, split, convolve_image, transform=None, target_transform=None,
                 keep_difficult=False, is_train=True, data_aug=None):

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        image_sets_file = os.path.join(dir_path, images_path)
        self.ids, self.labels = ImageDataset._read_image_ids(image_sets_file)
        self.class_names = open(os.path.join(dir_path, "imagenet-classes.txt"), 'r')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.convolve_image = convolve_image

        self.DATA_AUG = data_aug
        self.READ_FROM_OTHER_FOLDER = False

        logger = logging.getLogger("CDA.inference")
        logger.info("is Train   : {}".format(is_train))
        logger.info("Images Path: {}".format(image_sets_file))
        logger.info("Data Aug.  : {}".format(self.DATA_AUG))

    def __getitem__(self, index):
        image_id = self.ids[index]
        targets = self.labels[index]
        image = self._read_image(image_id)
        image = Image.fromarray(image)
        image = self.transform(image)

        return image, targets, image_id

    def __len__(self):
        return len(self.ids)

    @ staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        labels = []
        with open(image_sets_file) as f:
            for line in f:
                L = line.rstrip()
                L_ = L.split(" ")
                img_name = ' '.join(L_[:-1])
                ids.append(img_name)
                labels.append(int(L_[-1]))

        return ids, labels

    def _read_image(self, image_id):

        image_path = os.path.join(dir_path, '../../../', self.data_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        return image
