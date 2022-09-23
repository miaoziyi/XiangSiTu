import random
import pandas as pd
import time

n = 8
# df = pd.read_csv('..txt/result_1_few.csv')
df = pd.read_csv(r"D:\xiangsitu\related\imgs_new.csv")

leafCids = []
leafCids = df['leafCid'].unique()
print("个数：", len(leafCids))

for leafCid in leafCids:
    # print(time.strftime("%H:%M:%S", time.localtime()), leafCid)
    i = 0
    spuids = []
    spuids = df[df['leafCid'] == leafCid]['spuId']
    spuids = spuids.unique()
    nn = len(spuids)
    # 当三级目录商品不足时
    if nn < n:
        n = nn
    while i < n:
        index = random.randint(0, nn-1)
        spuid = spuids[index]
        df["skc"].loc[df["spuId"] == spuid] = '1'
        i += 1

df = df[df['skc'] == '1']
df.drop(['skc'], axis=1, inplace=True)
df.to_csv("../txt/per_leafCid_8"+".csv", index=False)
