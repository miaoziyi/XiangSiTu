import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm

error = []

all_images_meta = pd.read_csv('../txt/spuids_10000_1.csv')
all_images_meta = all_images_meta[['imgId', 'spuId', 'imgPath']]
print(all_images_meta.head())

print("Load dataset.................")
# images_meta_sample = all_images_meta
images_meta_sample = all_images_meta.groupby('spuId').head(2).reset_index(drop=True)

print("Form triplets.................")
landmark_groups = all_images_meta.groupby('spuId')
# print(landmark_groups.get_group(1))


def get_positive(anchor_landmark_id):
    landmark_group = landmark_groups.get_group(anchor_landmark_id)
    indexes = landmark_group.index.values
    #     print(indexes)

    rand_index = np.random.choice(indexes)
    pos_img_id = landmark_group.loc[rand_index].imgId

    return pos_img_id

# 修改，除去本身的正样本
# def get_positive(anchor_landmark_id, imgId):
#     landmark_group = landmark_groups.get_group(anchor_landmark_id)
#     # indexes = landmark_group.index.values
#     ids = landmark_group.imgId.values
#     ids = list(ids)
#     if imgId in ids and len(ids) != 1:
#         ids.remove(imgId)
#     # print(imgId)
#     # rand_index = np.random.choice(ids)
#     pos_img_id = np.random.choice(ids)
#
#     return pos_img_id

def get_negative(landmark_id):
    indexes = images_meta_sample.index.values

    for i in range(len(images_meta_sample)):
        rand_index = np.random.choice(indexes)

        neg_img_id = images_meta_sample.loc[rand_index].imgId
        neg_landmark_id = images_meta_sample.loc[rand_index].spuId

        if neg_landmark_id != landmark_id:
            return neg_img_id

    return neg_img_id


images_meta_sample['positive_id'] = images_meta_sample['spuId'].apply(get_positive)
# images_meta_sample['positive_id'] = images_meta_sample.apply(lambda x: get_positive(x['spuId'], x['imgId']), axis=1)
images_meta_sample['negative_id'] = images_meta_sample['spuId'].apply(get_negative)

images_meta_sample = images_meta_sample.rename({'imgId': 'anchor_id'}, axis='columns')
print(images_meta_sample.head(20))

print("Write to tfrecords...........")


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image(img_id):
    imgPath = all_images_meta[all_images_meta['imgId'] == img_id]['imgPath'].values[0]
    path = 'D:/xiangsitu/imgs/' + imgPath
    # img = all_images_meta[all_images_meta['imgId'] == img_id]
    # imgPath = str(img['cid1'].values[0]) + '/' + str(img['cid2'].values[0]) + '/' + str(img['leafCid'].values[0]) + '/' + str(img['img'].values[0]).split('/')[-1]
    # path = r'D:\我的东西\实验室\老师工作安排\图像处理项目\jf_data_new\imgs/' + imgPath
    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, size=(128, 128), method='nearest')
    image = tf.image.convert_image_dtype(image, tf.uint8)

    image = tf.image.encode_jpeg(image, quality=94, optimize_size=True)

    return image


def serialize_example(example):
    #     print(example.anchor_id)
    anchor_img = get_image(example.anchor_id)
    #     print(example.positive_id)
    positive_img = get_image(example.positive_id)
    negative_img = get_image(example.negative_id)

    feature = {
        'anchor_img': _bytes_feature(anchor_img),
        'positive_img': _bytes_feature(positive_img),
        'negative_img': _bytes_feature(negative_img),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


image_indexes = images_meta_sample.index.values
print(image_indexes)

# 15个文件
file_size = len(image_indexes) // 15
file_count = len(image_indexes) // file_size + int(len(image_indexes) % file_size != 0)
print("每个输出文件的大小：", file_size)


def write_tfrecord_file(file_index, file_size, image_indexes, error):
    with tf.io.TFRecordWriter('train%.2i.tfrec' % (file_index)) as writer:
        start = file_size * file_index
        end = file_size * (file_index + 1)

        for i in tqdm(image_indexes[start:end]):
            try:
                example = serialize_example(
                    images_meta_sample.loc[i]

                )
                writer.write(example)
            except:
                error.append(i)
                break


for file_index in range(file_count):
    print('Writing TFRecord %i of %i...'%(file_index, file_count))
    write_tfrecord_file(file_index, file_size, image_indexes, error)

np.save('error_10000', error)
