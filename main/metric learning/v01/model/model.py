import sys

sys.path.append('../../../../pytorch-image-models-master/')

from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# Visuals and CV2
import cv2

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from model.loss import ArcMarginProduct
import traceback

s = traceback.extract_stack()

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ImgNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=True,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ImgNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features

        # 后续可以改主干网络
        # self.backbone.blocks[6][0].conv_pwl = nn.Conv2d(1392, 566, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.backbone.blocks[6][0].bn3 = nn.BatchNorm2d(566, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.backbone.blocks[6][1].conv_pw = nn.Conv2d(566, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.backbone.blocks[6][1].conv_pwl = nn.Conv2d(2304, 566, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.backbone.blocks[6][1].bn3 = nn.BatchNorm2d(566, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.backbone.conv_head = nn.Conv2d(566, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # print(self.backbone)
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        # 加入空间注意力机制
        self.spatialattention = SpatialAttention()

        # 全连接fc之前的操作
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        if 'train' in s[0][0]:
            return logits
        elif 'infer' in s[0][0] or 'find' in s[0][0]:
            return feature, logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        x = self.backbone(x)
        # print(x.shape)
        # 添加空间注意力机制
        x = x * self.spatialattention(x)
        x = self.pooling(x).view(batch_size, -1)
        # print(x.shape)
        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x
