import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from .densenet import densenet121
import torchvision
from typing import Union, List, Dict, Any, cast


class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained, act_layer=-1, act_layer_mean=False):

        super(DenseNet121, self).__init__()

        #self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        self.densenet121 = densenet121(
            pretrained=isTrained, act_layer=act_layer, act_layer_mean=act_layer_mean)

        kernelCount = self.densenet121.classifier.in_features

        # CHANGED
        #self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount))  # removed softmax

    def forward(self, x, return_feats=False):

        if return_feats:
            x, feat = self.densenet121(x, return_feats)
            return x, feat

        else:
            x = self.densenet121(x)
            return x

    def get_feats(self, x):

        x, feat = self.densenet121.get_feats(x)
        return x, feat


class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet169, self).__init__()

        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features

        self.densenet169.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet169(x)
        return x


class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet201, self).__init__()

        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features

        self.densenet201.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet201(x)
        return x


# def DenseNet121(pretrained: bool = False, progress: bool = True, **kwargs):

#     return DenseNet121(14, True).cuda()
