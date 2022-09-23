import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa

import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
from tqdm.notebook import tqdm

import faiss
from faiss import normalize_L2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

BATCH_SIZE = 64 * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = 1451645 // BATCH_SIZE // 8
RATE = 0.0001

IMAGE_SIZE = 128
EMBED_SIZE = 256


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
    # r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\embedding_model10.h5',
    'model/embedding_model_spuid_1w_1.h5',
    custom_objects={'GeMPoolingLayer': GeMPoolingLayer}
)
# embedding_model = embedding_model.to(device)
print('load model:success.........')

image_ids = pd.read_csv(
    # r'D:\mzy\mzy\img1_0\train\data.csv'
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


images = [get_image(img) for img in image_ids.imgPath]
print('get image:success............')
images = np.array(images)

embeddings = embedding_model.predict(images)
np.save('data/data_embedding_1w_1.npy', embeddings)
