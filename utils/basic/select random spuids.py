import pandas as pd
import random

# 随机选择指定数量的spuId
df = pd.read_csv(r"D:\xiangsitu\related\imgs_new.csv")
i = 0
spuIds = 10000
while i < spuIds:
    spu = random.randint(1, 557976)
    print(spu)
    try:
        df["skc"].loc[df["spuId"] == spu] = '1'
        # df.loc[df['spuId'] == spu]['x'] = 1  会报错
        i += 1
    except:
        continue
df = df[df['skc'] == '1']
df.drop(['skc'], axis=1, inplace=True)
df.to_csv("txt/spuIds_"+str(spuIds)+".csv", index=False)
