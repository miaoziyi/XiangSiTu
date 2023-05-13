import sys
sys.path.append('../../../pytorch-image-models-master/')

from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np
from torch.nn import Parameter
# Visuals and CV2
import cv2

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

#torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader


import gc
import matplotlib.pyplot as plt

from data import infer_ImgDataset
from model import model as _ImgNet
# import cudf
# import cuml
# import cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml import PCA
# from cuml.neighbors import NearestNeighbors

import argparse

# 1. 定义命令行解析器对象
parser = argparse.ArgumentParser(description='Demo of argparse')

# 2. 添加命令行参数
parser.add_argument('--dimension', type=int, default=512)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('-m', '--model_path', required=True, help='训练模型路径')
parser.add_argument('-t', '--test_csv_path', required=True, help='测试集文件路径')
parser.add_argument('-i', '--test_image_path', required=True, help='测试集图片路径')

args = parser.parse_args()

NUM_WORKERS = 0
BATCH_SIZE = args.batch
# SEED = 2020

device = torch.device('cuda')

# 注意修改！！！！！！！！
CLASSES = 12039
# CLASSES = 145

################################################  ADJUSTING FOR CV OR SUBMIT ##############################################
# CHECK_SUB = False
# GET_CV = True
################################################# MODEL ####################################################################

model_name = 'efficientnet_b3' #efficientnet_b0-b7

################################################ MODEL PATH ###############################################################

# 修改
epoch = 50
weidu = args.dimension
DIM = (512, 512)
IMG_MODEL_PATH = args.model_path
# IMG_MODEL_PATH = r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\model_efficientnet_b3_IMG_SIZE_512_arcface (1).bin'

################################################ Metric Loss and its params #######################################################
# 针对img test
loss_module = 'arcface' # 'cosface' #'adacos' arcface

# 针对img_1w训练集
# loss_module = IMG_MODEL_PATH.split('_')[6]
# 修改
# s = 30
s = 15
# m = 0.5
ls_eps = 0.0
easy_margin = False


def get_test_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )


def get_image_embeddings(image_paths):
    embeds = []

    model = _ImgNet.ImgNet(n_classes=CLASSES, model_name=model_name)
    model.eval()

    model.load_state_dict(torch.load(IMG_MODEL_PATH), strict=False)
    model = model.to(device)

    image_dataset = infer_ImgDataset.ImgDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for img, label in tqdm(image_loader):
            img = img.cuda()
            label = label.cuda()
            feat, _ = model(img, label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model

    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


# 修改 更清楚的测试集命名
df = pd.read_csv(args.test_csv_path)
image_paths = args.test_image_path + df['imgPath']
image_embeddings = get_image_embeddings(image_paths.values)

# 修改
name = args.model_path.replace('.bin', '')
np.save(f'embedding/{name}.npy', image_embeddings)
print(f'测试集嵌入向量保存至embedding/{name}.npy')
