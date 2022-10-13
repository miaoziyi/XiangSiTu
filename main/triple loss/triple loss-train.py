import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
from tqdm.notebook import tqdm
# from kaggle_datasets import KaggleDatasets

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')

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

# PT1_GCS_DS_PATH = KaggleDatasets().get_gcs_path('results')
# print(PT1_GCS_DS_PATH)
data_dir = '../tfrecords/spuids_1w'

BATCH_SIZE = 64 * strategy.num_replicas_in_sync
EPOCHS = 20
STEPS_PER_EPOCH = 1451645 // BATCH_SIZE // 8
RATE = 0.0001

IMAGE_SIZE = 128
EMBED_SIZE = 256

filenames = tf.io.gfile.glob([
    data_dir + '/*.tfrec'
])
print('--------')
print(len(filenames))

train_data = tf.data.TFRecordDataset(
    filenames,
    num_parallel_reads=tf.data.experimental.AUTOTUNE
)
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False
train_data = train_data.with_options(ignore_order)


def get_triplet(example):
    tfrec_format = {
        "anchor_img": tf.io.FixedLenFeature([], tf.string),
        "positive_img": tf.io.FixedLenFeature([], tf.string),
        "negative_img": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, tfrec_format)

    x = {
        'anchor_input': decode_image(example['anchor_img']),
        'positive_input': decode_image(example['positive_img']),
        'negative_input': decode_image(example['negative_img']),
    }

    return x, [0, 0, 0]


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method='nearest')

    image = augment(image)

    return image


def augment(image):
    rand_aug = np.random.choice([0, 1, 2, 3])

    if rand_aug == 0:
        image = tf.image.random_brightness(image, max_delta=0.3)
    elif rand_aug == 1:
        image = tf.image.random_contrast(image, lower=0.2, upper=0.4)
    elif rand_aug == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
    else:
        image = tf.image.random_saturation(image, lower=0.2, upper=0.4)

    rand_aug = np.random.choice([0, 1, 2, 3])

    if rand_aug == 0:
        image = tf.image.random_flip_left_right(image)
    elif rand_aug == 1:
        image = tf.image.random_flip_up_down(image)
    elif rand_aug == 2:
        rand_rot = np.random.randn() * 45
        image = tfa.image.rotate(image, rand_rot)
    else:
        image = tfa.image.transform(image, [1.0, 1.0, -50, 0.0, 1.0, 0.0, 0.0, 0.0])

    image = tf.image.random_crop(image, size=[100, 100, 3])
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    return image


train_data = train_data.map(
    get_triplet,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
train_data = train_data.repeat()
train_data = train_data.shuffle(1024)
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

# fig, axes = plt.subplots(5, 3, figsize=(15, 15))

for images, landmark_id in train_data.take(1):
    anchors = images['anchor_input']
    positives = images['positive_input']
    negatives = images['negative_input']


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


reg = tf.keras.regularizers

with strategy.scope():
    # backbone
    backbone = tf.keras.applications.Xception(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights='imagenet',
        include_top=False
    )

    backbone.trainable = False

    # embedding model
    x_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = backbone(x_input)
    x = GeMPoolingLayer()(x)
    x = tf.keras.layers.Dense(EMBED_SIZE, activation='softplus', kernel_regularizer=reg.l2(), dtype='float32')(x)

    embedding_model = tf.keras.models.Model(inputs=x_input, outputs=x, name="embedding")

    # anchor encoding
    anchor_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='anchor_input')
    anchor_x = embedding_model(anchor_input)

    # positive encoding
    positive_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='positive_input')
    positive_x = embedding_model(positive_input)

    # anchor encoding
    negative_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='negative_input')
    negative_x = embedding_model(negative_input)

    # construct model
    model = tf.keras.models.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_x, positive_x, negative_x]
    )

embedding_model.summary()


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchors = y_pred[0]
    positives = y_pred[1]
    negatives = y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchors, positives)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchors, negatives)), axis=-1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=RATE),
    loss=triplet_loss
)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
]
history = model.fit_generator(
    train_data,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
)

plt.title('Model loss')
plt.plot(history.history['loss'])
plt.show()


def distance_test(anchors, positives, negatives):
    pos_dist = []
    neg_dist = []

    anchor_encodings = embedding_model.predict(anchors)
    positive_encodings = embedding_model.predict(positives)
    negative_encodings = embedding_model.predict(negatives)

    for i in range(len(anchors)):
        pos_dist.append(
            distance.euclidean(anchor_encodings[i], positive_encodings[i])
        )
        neg_dist.append(
            distance.euclidean(anchor_encodings[i], negative_encodings[i])
        )

    return pos_dist, neg_dist


pos_dist, neg_dist = distance_test(anchors[0:5], positives[0:5], negatives[0:5])
fig, axes = plt.subplots(5, 3, figsize=(15, 20))
for i in range(5):
    axes[i, 0].set_title('Anchor')
    axes[i, 0].imshow(anchors[i])

    axes[i, 1].set_title('Positive dist: {:.2f}'.format(pos_dist[i]))
    axes[i, 1].imshow(positives[i])

    axes[i, 2].set_title('Negative dist: {:.2f}'.format(neg_dist[i]))
    axes[i, 2].imshow(negatives[i])
plt.show()

embedding_model.save(
    'model/embedding_model_spuid_1w_1.h5',
    save_format='h5',
    overwrite=True
)
