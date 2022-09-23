import pandas as pd
from PIL import Image
import os

ff = open('../txt/invalid_imgIds.txt', 'w')

data = pd.read_csv('../txt/per_leafCid_250_StratifiedKFold.csv')
print(data)
ll = len(data)
for i in range(ll):
    if i % 1000 == 0:
        print(i)
    path = 'D:/xiangsitu/imgs/' + data.loc[i]['imgPath']
    try:
        img_array = Image.open(path).convert("RGB")
    except:
        ff.write(str(data.loc[i]['imgId']) + '\n')
        data = data.drop(i)
        os.remove(path)

data.to_csv('../txt/per_leafCid_250_StratifiedKFold.csv', index=False)

# data = pd.read_csv('../txt/per_leafCid_250_StratifiedKFold.csv')
# error = [424354, 1441551, 2378338, 3308543, 3333400, 3333715, 3337892, 4239484, 4258290, 4746572, 5079368]
# ids = list(data.imgId)
# ids_valid = []
# for e in ids:
#     if e not in error:
#         ids_valid.append(e)
# data = data[data.imgId.isin(ids_valid)]
# data.to_csv('../txt/per_leafCid_250_StratifiedKFold_valid.csv', index=False)
