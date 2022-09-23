# import torch
# import torchvision
#
# print(torchvision.__version__)
# print(torch.__version__)

# print('../input/imgs-1-cid2/imgs_1_cid2/11/1101/20220317201951828.jpg'.split('/',5)[5])
# from PIL import Image
#
# try:
#     img_array = Image.open('D:/xiangsitu/imgs/12/1201/120199/20220415044001285119940.jpg').convert("RGB")
# except:
#     print(0)

# s = '92182,13990,12,1201,120101,images/crawler/20220318/20220318003038885.jpg,12/1201/120101/20220318003038885.jpg'
# x = s.split(',')
# print(x[2])
import cv2
import numpy as np
import faiss
from faiss import normalize_L2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_SIZE = 128

# Loads the embedding
embeddings = np.load('data/data_embedding_f2.npy')
print('load embeddings:success.........')
print(type(embeddings))
print(embeddings.shape)
normalize_L2(embeddings)

# faiss索引构建
index = faiss.IndexIDMap(faiss.IndexFlatIP(2048))
index.add_with_ids(embeddings, np.array(range(0, len(embeddings))).astype('int64'))
faiss.write_index(index, 'imgs.index')
print('create faiss index:success.........')

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


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(
            inputs,
            clip_value_min=self.eps,
            clip_value_max=tf.reduce_max(inputs)
        )
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1. / self.p)

        return inputs

    def get_config(self):
        return {
            'p': self.p,
            'eps': self.eps
        }

# 用保存好的模型
embedding_model = tf.keras.models.load_model(
    r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\embedding_model.h5',
    custom_objects={'GeMPoolingLayer': GeMPoolingLayer}
)
print('load model:success.........')

# 获取预测图片嵌入
# img = [get_image2(r'D:\xiangsitu\img_test\17\1702\170201\20220321170250544.jpg')]

# print('load predict image:success.........')

import time
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


# print('similar_images:', similar_images)

image_ids = pd.read_csv(
    'txt/img_test_path.csv'
)
print('read csv:success...........')

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

# 保存搜索结果图片
with open('txt/query-imgPath2.txt', 'r', encoding='utf8') as file:
    imgs = file.readlines()
    for m, img in enumerate(imgs):
        print(m)
        img2 = get_image2(img.replace('\n', ''))
        img = [img2]
        fig, ax = plt.subplots(7, 8, figsize=(50, 50))
        ax[0, 0].imshow(img2)

        img = np.array(img)
        embedding = embedding_model.predict(img)
        similar_images = search(embedding, top_k=49, index=index)

        for i in range(7):
            for j in range(1, 8):
                img_index = similar_images[j - 1 + (i * 7)]

                landmark_id = image_ids.loc[img_index].spuId
            #         dist = distances[3999, img_index]
            #         ax[i,j].set_title('spuId: {}, dist: {:.2f}'.format(landmark_id, dist))
            #     ax[i, j].set_title('spuId: {}'.format(img_index))
                ax[i, j].imshow(
                    get_image(image_ids.loc[img_index].imgPath)
                )
        plt.savefig(str(m) + '.jpg')
        plt.close()
