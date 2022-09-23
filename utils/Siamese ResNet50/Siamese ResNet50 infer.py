import pandas as pd
import tensorflow as tf
import keras_toolkit as kt
import faiss
from faiss import normalize_L2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

target_shape = (256, 256)
IMAGE_SIZE = 256

def preprocess_image(filename, target_shape=target_shape):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    img_str = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_str, channels=3)
    img = tf.image.resize(img, target_shape)

    # Resnet-style preprocessing, see: https://git.io/JYo77
    mean = [103.939, 116.779, 123.68]
    img = img[..., ::-1]
    img -= mean

    return img


strategy = kt.accelerator.auto_select(verbose=True)
with strategy.scope():
    encoder = tf.keras.models.load_model(
       'model/encoder.h5'
    )

imgs = pd.read_csv('../../utils/txt/img_test_path.csv')
dimgs = kt.image.build_dataset(
    'D:/xiangsitu/img_test/' + imgs['imgPath'],
    decode_fn=preprocess_image
)

# predict
embeddings = encoder.predict(dimgs, verbose=1)
# 记得修改
np.save('../data/Siamese ResNet50/siamese_resnet_250_1.npy', embeddings)
