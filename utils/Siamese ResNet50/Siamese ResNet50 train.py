from functools import partial

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from sklearn.model_selection import train_test_split
import keras_toolkit as kt

target_shape = (256, 256)


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


def build_triplets_dset(df, bsize=32, cache=True, shuffle=False):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    build_dataset = partial(
        kt.image.build_dataset,
        decode_fn=preprocess_image,
        bsize=bsize,
        cache=cache,
        shuffle=False
    )

    danchor = build_dataset(df.anchor)
    dpositive = build_dataset(df.positive)
    dnegative = build_dataset(df.negative)

    dset = tf.data.Dataset.zip((danchor, dpositive, dnegative))

    if shuffle:
        dset = dset.shuffle(shuffle)

    return dset


# COMPETITION_NAME = 'shopee-product-matching'
strategy = kt.accelerator.auto_select(verbose=True)
PATH = r'D:\xiangsitu\imgs/'
# 8
BATCH_SIZE = 8 * 16
# BATCH_SIZE = strategy.num_replicas_in_sync * 16

train = pd.read_csv('../txt/Siamese ResNet50/train_triplets_imgs.csv')

train = train.apply(lambda col: PATH + col)
train_paths, val_paths = train_test_split(train, train_size=0.8, random_state=42)
print(train_paths.head())

dtrain = build_triplets_dset(
    train_paths,
    bsize=BATCH_SIZE,
    cache=True,
    # 修改
    # shuffle=8192
    shuffle=False
)

dvalid = build_triplets_dset(
    val_paths,
    bsize=BATCH_SIZE,
    cache=True,
    shuffle=False
)


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


with strategy.scope():
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False, pooling='avg',
    )
    dropout = layers.Dropout(0.5, name='dropout')
    reduce = layers.Dense(512, activation='linear', name='reduce')

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        reduce(dropout(base_cnn(anchor_input))),
        reduce(dropout(base_cnn(positive_input))),
        reduce(dropout(base_cnn(negative_input))),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

siamese_network.summary()


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


with strategy.scope():
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))

hist = siamese_model.fit(dtrain, epochs=15, validation_data=dvalid)

with strategy.scope():
    encoder = tf.keras.Sequential([
        siamese_model.siamese_network.get_layer('resnet50'),
        siamese_model.siamese_network.get_layer('dropout'),
        siamese_model.siamese_network.get_layer('reduce'),
    ])

    encoder.save('model/encoder.h5')

siamese_model.save_weights('model/siamese_model.h5')
siamese_model.siamese_network.save_weights('model/siamese_network.h5')
siamese_model.siamese_network.get_layer('resnet50').save_weights('model/resnet50.h5')

import plotly.express as px

px.line(hist.history)
