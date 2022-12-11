import sys

sys.path.append('../../../pytorch-image-models-master/')

from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np

# Visuals and CV2
import cv2
import json
# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import gc
import matplotlib.pyplot as plt
import faiss
from faiss import normalize_L2
from PIL import Image
import time
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
parser.add_argument('--batch', type=int, default=24)
parser.add_argument('-m', '--model_path', required=True, help='训练模型路径')
parser.add_argument('-tc', '--test_csv_path', required=True, help='测试集文件路径')
parser.add_argument('-ta', '--test_ads_path', required=True, help='测试集绝对路径')
parser.add_argument('-tr', '--test_rel_path', required=True, help='测试集相对路径')
parser.add_argument('-qc', '--query_csv_path', required=True, help='搜索集文件路径')
parser.add_argument('-te', '--test_embedding_path', required=True, help='测试集嵌入向量路径')

args = parser.parse_args()

weidu = args.dimension
DIM = (512, 512)
# IMAGE_SIZE = 128
NUM_WORKERS = 0
BATCH_SIZE = args.batch
SEED = 2020
device = torch.device('cuda')

# 注意修改!!!!!!!!!!!!!!!!!
CLASSES = 11973
# CLASSES = 145

################################################  ADJUSTING FOR CV OR SUBMIT ##############################################
CHECK_SUB = False
GET_CV = True
################################################# MODEL ####################################################################
model_name = 'efficientnet_b3'  # efficientnet_b0-b7
################################################ MODEL PATH ###############################################################

# 模型修改
IMG_MODEL_PATH = args.model_path
# IMG_MODEL_PATH = r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\model_efficientnet_b3_IMG_SIZE_512_arcface (1).bin'
################################################ Metric Loss and its params #######################################################
loss_module = 'arcface'  # 'cosface' #'adacos'
s = 15.0
# s = 10.0
# s = 30.0
m = 0.5
ls_eps = 0.0
easy_margin = False


def get_test_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0], DIM[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )


def get_image_embeddings(image_paths):
    embeds = []

    model = _ImgNet.ImgNet(n_classes=CLASSES, model_name=model_name)
    model.eval()

    model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location='cuda'), strict=False)
    model = model.to(device)
    print(model.final)

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


def search(query_vector, top_k, index):
    t = time.time()
    print(query_vector.shape)
    #     print(query_vector)
    normalize_L2(query_vector)
    top_k = index.search(query_vector, top_k)
    #     print(top_k)
    print('>>>> Results in Total Time: {}'.format(time.time() - t))
    top_k_ids = top_k[1].tolist()[0]
    return top_k_ids

# 修改
embeddings = np.load(args.test_embedding_path)
normalize_L2(embeddings)
# faiss索引构建
index = faiss.IndexIDMap(faiss.IndexFlatIP(weidu))
index.add_with_ids(embeddings, np.array(range(0, len(embeddings))).astype('int64'))
# 保存索引
# faiss.write_index(index, 'imgs-1w.index')

# 修改
image_ids = pd.read_csv(args.test_csv_path)
print('read csv:success...........')

# 加载搜索集
df = pd.read_csv(args.query_csv_path)
image_paths = df['imgPath']
image_embeddings = get_image_embeddings(image_paths.values)

e = {}
with open(args.query_csv_path, 'r', encoding='utf8') as file:
    imgs = file.readlines()[1:]
for i in range(len(image_embeddings)):
    print(i)
    img_json = imgs[i].replace(args.test_ads_path, '').replace('\n', '')
    embedding = np.array([image_embeddings[i]])
    similar_images = search(embedding, top_k=49, index=index)
    query_images = []
    for i in range(7):
        for j in range(7):
            img_index = similar_images[j + (i * 7)]
            query_images.append(args.test_rel_path + image_ids.loc[img_index].imgPath)
    e[img_json] = query_images

# 修改
name = args.model_path.replace('.bin', '')
with open(f"{name}.json", 'w', encoding='utf-8') as f:
    json.dump(e, f)
print(f'json文件保存在{name}.json')
