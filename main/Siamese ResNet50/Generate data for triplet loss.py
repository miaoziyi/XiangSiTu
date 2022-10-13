import random

import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


def generate_triplets(df):
    # Source: https://www.kaggle.com/xhlulu/shopee-generate-data-for-triplet-loss
    random.seed()
    group2df = dict(list(df.groupby('spuId')))

    def aux(row):
        anchor = row.imgId

        # We sample a positive data point from the same group, but
        # exclude the anchor itself 正样本除去本身
        ids = group2df[row.spuId].imgId.tolist()
        if len(ids) != 1:
            ids.remove(row.imgId)
        positive = random.choice(ids)

        # Now, this will sample a group from all possible groups, then sample
        # a product from that group 负样本
        groups = list(group2df.keys())
        groups.remove(row.spuId)
        neg_group = random.choice(groups)
        negative = random.choice(group2df[neg_group].imgId.tolist())

        return anchor, positive, negative
    return aux


train = pd.read_csv('../img1_0/img-300w/per_leafCid_250_1.csv')

# Useful dictionaries; use below to convert if needed
id_to_img = train.set_index('imgId').imgPath.to_dict()
# id_to_title = train.set_index('imgId').title.to_dict()

train_triplets = train.progress_apply(generate_triplets(train), axis=1).tolist()
train_triplets_df = pd.DataFrame(train_triplets, columns=['anchor', 'positive', 'negative'])
print(train_triplets_df.head(20))

train_triplets_imgs = train_triplets_df.applymap(lambda x: id_to_img[x])
print(train_triplets_imgs.head())

train_triplets_imgs.to_csv('txt/train_triplets_imgs.csv', index=False)
# train_triplets_titles.to_csv('train_triplets_titles.csv', index=False)
# train_triplets_df.to_csv('train_triplets_ids.csv', index=False)
