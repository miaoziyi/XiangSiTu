import cv2
import numpy as np
import faiss
from faiss import normalize_L2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from annoy import AnnoyIndex
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
IMAGE_SIZE = 128

image_ids = pd.read_csv(
    r'D:\mzy\mzy\img1_0\train\result_1.csv'
)

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


# Loads the embedding
embeddings = np.load('data/data_embedding_f.npy')
print('shape:', embeddings.shape)
t = AnnoyIndex(2048, metric='euclidean')

ntree = 50

for i, vector in enumerate(embeddings):
    t.add_item(i, vector)
_ = t.build(ntree)

# 用保存好的模型
embedding_model = tf.keras.models.load_model(
    r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\embedding_model.h5',
    custom_objects={'GeMPoolingLayer': GeMPoolingLayer}
)

def get_similar_images_annoy2(embedding):
    start = time.time()
    # base_img_id, base_label  = df_new.iloc[img_index, [0, 1]]
    # print(base_img_id)
    similar_img_ids = t.get_nns_by_vector(embedding[0], 49)
    # similar_img_ids = t.get_nns_by_vector(activation['fc2'][0].tolist(), 8)
    # print(similar_img_ids)
    end = time.time()
    print(f'{(end - start) * 1000} ms')
    return similar_img_ids

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

with open('txt/query-imgPath.txt', 'r', encoding='utf8') as file:
    imgs = file.readlines()
    for m, img in enumerate(imgs):
        print(m)
        img2 = get_image2(img.replace('\n', ''))
        img = [img2]
        fig, ax = plt.subplots(7, 8, figsize=(50, 50))
        ax[0, 0].imshow(img2)

        img = np.array(img)
        embedding = embedding_model.predict(img)
        similar_images = get_similar_images_annoy2(embedding)

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
    # plt.show()
