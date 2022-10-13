import numpy as np  # linear algebra
import pandas as pd
import cv2, matplotlib.pyplot as plt
import os

train = pd.read_csv('../img1_0/img-300w/per_leafCid_250_0_3.csv', keep_default_na=False)
print('train shape is', train.shape)
# train = train[train['imgPath'] != '']
print('train shape is', train.shape)
print(train[:5])

BASE = 'D:/xiangsitu/imgs/'


def displayDF(train, random=False, COLS=10, ROWS=20, path=BASE):
    for k in range(ROWS):
        plt.figure(figsize=(20, 5))
        for j in range(COLS):
            if random:
                row = np.random.randint(0, len(train))
            else:
                row = COLS * k + j
            img_path = str(train.iloc[row]['cid1']) + '/' + str(train.iloc[row]['cid2']) + '/' + str(train.iloc[row]['leafCid']) + '/' + train.iloc[row]['img'].split('/')[-1]
            print(img_path)
            #             name =
            #             title = train.iloc[row,3]
            #             title_with_return = ""
            #             for i,ch in enumerate(title):
            #                 title_with_return += ch
            #                 if (i!=0)&(i%20==0): title_with_return += '\n'
            if os.path.exists(path + img_path):
                img = cv2.imread(path + img_path)
                plt.subplot(1, COLS, j + 1)
                plt.title(train.iloc[row, 1])
                plt.axis('off')
                plt.imshow(img)
        plt.show()


displayDF(train, random=True)
