import pandas as pd
import os


# 判断路径是否存在
def get_path(cid1, cid2, leafCid, img):
    path = str(cid1) + '/' + str(cid2) + '/' + str(leafCid) + '/' + img
    if os.path.exists(r'D:\xiangsitu\imgs/' + path):
        return path
    else:
        return ""


train_df = pd.read_csv('../txt/per_leafCid_250_StratifiedKFold.csv', keep_default_na=False)
train_df['imgPath'] = train_df.apply(lambda x: get_path(x.cid1, x.cid2, x.leafCid, x.img.split('/')[-1]), axis=1)

train_df = train_df[train_df["imgPath"] != ""]
train_df.to_csv("../txt/per_leafCid_250_StratifiedKFold.csv", index=False)