import logging
import os

import torch
import torch.utils.data
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image

from ssd.data.build import make_data_loader
from ssd.data.datasets.evaluation import evaluate

from ssd.utils import dist_util, mkdir
from ssd.utils.dist_util import synchronize, is_main_process
from vizer.draw import draw_boxes
from ssd.data.datasets import COCODataset, VOCDataset

class_names = VOCDataset.class_names


def generate_mask(boxes, images):

    mask = np.zeros((images.shape[2:]))
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        mask[y1:y2, x1:x2] = 1
    return torch.tensor(mask).unsqueeze(0).unsqueeze(0).cuda()


def entropy_loss(x):
    out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    out = -1.0 * out.sum(dim=1)
    return out.mean()


def add_mean(img, mean):

    out = img.clone()

    out[:, 0] = img[:, 0] + mean[0]
    out[:, 1] = img[:, 1] + mean[1]
    out[:, 2] = img[:, 2] + mean[2]
    return out


def subtract_mean(img, mean):

    out = img.clone()

    out[:, 0] = img[:, 0] - mean[0]
    out[:, 1] = img[:, 1] - mean[1]
    out[:, 2] = img[:, 2] - mean[2]
    return out


def preprocess(img, mean):

    img = torch.mul(img, 255.0)
    # out = img.clone()

    img[:, 0] = img[:, 0] - mean[0]
    img[:, 1] = img[:, 1] - mean[1]
    img[:, 2] = img[:, 2] - mean[2]

    # print(torch.max(out))
    return img


def normalize_to_minus_1_to_plus_1(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor
