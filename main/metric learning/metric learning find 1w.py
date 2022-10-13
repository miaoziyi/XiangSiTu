import sys

sys.path.append('../../pytorch-image-models-master/')

from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np
from torch.nn import Parameter
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
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import gc
import matplotlib.pyplot as plt
import faiss
from faiss import normalize_L2
from PIL import Image
import time
# import cudf
# import cuml
# import cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml import PCA
# from cuml.neighbors import NearestNeighbors

DIM = (512, 512)
IMAGE_SIZE = 128
NUM_WORKERS = 0
BATCH_SIZE = 24
SEED = 2020

device = torch.device('cuda')

# 注意修改!!!!!!!!!!!!!!!!!
CLASSES = 2079

################################################  ADJUSTING FOR CV OR SUBMIT ##############################################
CHECK_SUB = False
GET_CV = True
################################################# MODEL ####################################################################
model_name = 'efficientnet_b3'  # efficientnet_b0-b7
################################################ MODEL PATH ###############################################################

# 模型修改
IMG_MODEL_PATH = '../model/metric 17/model_efficientnet_b3_IMG_SIZE_512_arcface_50_17_bigS.bin'
# IMG_MODEL_PATH = r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\model_efficientnet_b3_IMG_SIZE_512_arcface (1).bin'

################################################ Metric Loss and its params #######################################################
epoch = IMG_MODEL_PATH.split('_')[7]
# 在这个里面好像不影响
# loss_module = 'cosface' # 'cosface' #'adacos' arcface
loss_module = IMG_MODEL_PATH.split('_')[6]

s = 30.0
m = 0.5
ls_eps = 0.0
easy_margin = False


class ImgDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(1)


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi / 4):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s

        return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ImgNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=True,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module=loss_module,
                 # loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=False):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ImgNet, self).__init__()
        print('Model building for {} backbone'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

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
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
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
        return feature, logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x


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

    model = ImgNet(n_classes=CLASSES, model_name=model_name)
    model.eval()

    model.load_state_dict(torch.load(IMG_MODEL_PATH, map_location='cuda'), strict=False)
    model = model.to(device)
    print(model.final)

    image_dataset = ImgDataset(image_paths=image_paths, transforms=get_test_transforms())
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


def get_image(img):
    image = Image.open('D:/xiangsitu/img_test/' + img)
    try:
        layers = image.layers
        if layers == 3:
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.asarray(image) / 255.0
            return image
    except:
        I1 = cv2.imread('D:/xiangsitu/img_test/' + img)
        I1_cvt2pil = Image.fromarray(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
        I1_cvt2pil = I1_cvt2pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        I1_cvt2pil = np.asarray(I1_cvt2pil) / 255.0
        return I1_cvt2pil


def get_image2(img):
    image = Image.open(img)
    try:
        layers = image.layers
        if layers == 3:
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.asarray(image) / 255.0
            return image
    except:
        I1 = cv2.imread(img)
        I1_cvt2pil = Image.fromarray(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
        I1_cvt2pil = I1_cvt2pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        I1_cvt2pil = np.asarray(I1_cvt2pil) / 255.0
        return I1_cvt2pil


# 修改
embeddings = np.load(f'../data/metric 17/metric_learning_17_{epoch}_{loss_module}_bigS.npy')
normalize_L2(embeddings)
# faiss索引构建
index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
index.add_with_ids(embeddings, np.array(range(0, len(embeddings))).astype('int64'))
# 保存索引
# faiss.write_index(index, 'imgs-1w.index')

# 加载搜索集
df = pd.read_csv('../txt/query test/query_1w_imgPath.csv')
image_paths = df['imgPath']
image_embeddings = get_image_embeddings(image_paths.values)

def search(query_vector, top_k, index):
    t = time.time()
    #     print(query_vector.shape)
    #     print(query_vector)
    normalize_L2(query_vector)
    top_k = index.search(query_vector, top_k)
    #     print(top_k)
    print('>>>> Results in Total Time: {}'.format(time.time() - t))
    top_k_ids = top_k[1].tolist()[0]
    return top_k_ids


def get_image(img):
    image = Image.open('D:\mzy\imgs_1w\imgs_1/' + img)
    image = image.resize((128, 128))
    image = np.asarray(image) / 255.0
    return image


image_ids = pd.read_csv(
    r'D:\mzy\imgs_1w/result_1.csv'
)
for n in range(len(image_embeddings)):
    print(n)
    similar_images = search(np.array([image_embeddings[n]]), top_k=64, index=index)

    fig, ax = plt.subplots(8, 8, figsize=(60, 60))
    for i in range(8):
        for j in range(8):
            img_index = similar_images[j + (i * 8)]

            landmark_id = image_ids.loc[img_index].spuId
            ax[i, j].set_title('spuId: {}'.format(img_index))
            ax[i, j].imshow(
                get_image(image_ids.loc[img_index].imgPath)
            )
    # plt.show()
    fig.savefig(r'D:\mzy\imgs_1w\query\result/'+str(n+1)+'_'+str(epoch)+'_'+str(DIM[0])+'_'+loss_module+'_bigS'+'.png')
