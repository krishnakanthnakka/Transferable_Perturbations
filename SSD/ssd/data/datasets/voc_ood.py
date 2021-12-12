import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import logging

from ssd.structures.container import Container


class VOCDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        #self.data_dir = '/cvlabdata2/home/krishna/Datasets/VOC/VOCdevkit/VOC2007'

        image_sets_file = os.path.join(
            self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        logger = logging.getLogger("SSD.Dataloader")

        logger.info("Size:{}, fpath:{}".format(
            len(self.ids), image_sets_file))
        # exit()

        self.class_dict = {class_name: i for i,
                           class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)

        #print("before", boxes.shape)
        # exit()

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]

        #print("After", boxes.shape)
        # exit()

        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        # print("KK", boxes)
        # exit()

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        #print("after2", boxes.shape)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )

        #print(targets['boxes'].shape, torch.sum(targets['labels'] > 0))
        # exit()

        return image, targets, image_id

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(
            self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # print(class_name)
            # exit()

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str)
                                if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(
            self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(
            map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(
            self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
