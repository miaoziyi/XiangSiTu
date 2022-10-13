import json
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
print('load encoder:success...')

embeddings = np.load('../data/Siamese ResNet50/siamese_resnet_250_1.npy', )
print('load test embeddings:success...')
normalize_L2(embeddings)

# faiss索引构建
index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
index.add_with_ids(embeddings, np.array(range(0, len(embeddings))).astype('int64'))


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


test = pd.read_csv('../txt/query-imgPath2.csv')
print('read query csv:success...')
dtest = kt.image.build_dataset(
    test['imgPath'],
    decode_fn=preprocess_image
)

# predict
test_embeddings = encoder.predict(dtest, verbose=1)
print('query embeddings predict:success...')

image_ids = pd.read_csv('../txt/img_test_path.csv')
e = {}
with open('../txt/query-imgPath_old.txt', 'r', encoding='utf8') as file:
    imgs = file.readlines()
for i in range(len(test_embeddings)):
    print(i)
    img_json = imgs[i].replace('D:/xiangsitu/', '').replace('\n', '')
    embedding = np.array([test_embeddings[i]])
    similar_images = search(embedding, top_k=49, index=index)
    query_images = []
    for i in range(7):
        for j in range(7):
            img_index = similar_images[j + (i * 7)]
            query_images.append("img_test/" + image_ids.loc[img_index].imgPath)
    e[img_json] = query_images
with open("D:/xiangsitu/m/siamese_query250_1.json", 'w', encoding='utf-8') as f:
    json.dump(e, f)
