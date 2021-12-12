import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from ssd.structures.container import Container


class SIM10KDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'car')

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

        self.data_dir = '/cvlabdata1/home/krishna/DA/SSD/datasets/SIM10K'

        image_sets_file = os.path.join(
            self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        self.ids = SIM10KDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i,
                           class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]

        image = self._read_image(image_id)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        # print("KK", boxes)
        # exit()

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )

        return image, targets, index

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
            if not class_name == "car":
                continue

            bbox = obj.find('bndbox')
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


# import os

# import torch
# import torch.utils.data
# from PIL import Image
# import sys

# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET


# from .bounding_box import BoxList


# class SIM10KDataset(torch.utils.data.Dataset):

#     CLASSES = (
#         "__background__ ",
#         "car",
#     )

#     def __init__(self, data_dir, split, use_difficult=False, transforms=None):
#         self.root = data_dir
#         self.image_set = split
#         self.keep_difficult = use_difficult
#         self.transforms = transforms

#         self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
#         self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
#         self._imgsetpath = os.path.join(
#             self.root, "ImageSets", "Main", "%s.txt")

#         with open(self._imgsetpath % self.image_set) as f:
#             self.ids = f.readlines()
#         self.ids = [x.strip("\n") for x in self.ids]
#         self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

#         cls = Sim10kDataset.CLASSES
#         self.class_to_ind = dict(zip(cls, range(len(cls))))

#     def __getitem__(self, index):
#         img_id = self.ids[index]
#         img = Image.open(self._imgpath % img_id).convert("RGB")

#         target = self.get_groundtruth(index)
#         target = target.clip_to_image(remove_empty=True)

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target, index

#     def __len__(self):
#         return len(self.ids)

#     def get_groundtruth(self, index):
#         img_id = self.ids[index]
#         anno = ET.parse(self._annopath % img_id).getroot()
#         anno = self._preprocess_annotation(anno)

#         height, width = anno["im_info"]
#         target = BoxList(anno["boxes"], (width, height), mode="xyxy")
#         target.add_field("labels", anno["labels"])
#         target.add_field("difficult", anno["difficult"])
#         return target

#     def _preprocess_annotation(self, target):
#         boxes = []
#         gt_classes = []
#         difficult_boxes = []
#         TO_REMOVE = 1

#         for obj in target.iter("object"):
#             difficult = int(obj.find("difficult").text) == 1
#             if not self.keep_difficult and difficult:
#                 continue
#             name = obj.find("name").text.lower().strip()
#             # ignore if not car
#             if not name == "car":
#                 continue

#             bb = obj.find("bndbox")
#             # Make pixel indexes 0-based
#             # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
#             box = [
#                 bb.find("xmin").text,
#                 bb.find("ymin").text,
#                 bb.find("xmax").text,
#                 bb.find("ymax").text,
#             ]
#             bndbox = tuple(
#                 map(lambda x: x - TO_REMOVE, list(map(int, box)))
#             )

#             boxes.append(bndbox)
#             gt_classes.append(self.class_to_ind[name])
#             difficult_boxes.append(difficult)

#         size = target.find("size")
#         im_info = tuple(
#             map(int, (size.find("height").text, size.find("width").text)))

#         res = {
#             "boxes": torch.tensor(boxes, dtype=torch.float32),
#             "labels": torch.tensor(gt_classes),
#             "difficult": torch.tensor(difficult_boxes),
#             "im_info": im_info,
#         }
#         return res

#     def get_img_info(self, index):
#         img_id = self.ids[index]
#         anno = ET.parse(self._annopath % img_id).getroot()
#         size = anno.find("size")
#         im_info = tuple(
#             map(int, (size.find("height").text, size.find("width").text)))
#         return {"height": im_info[0], "width": im_info[1]}

#     def map_class_id_to_class_name(self, class_id):
#         return Sim10kDataset.CLASSES[class_id]
