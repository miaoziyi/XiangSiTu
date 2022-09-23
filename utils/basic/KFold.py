# 简单的k folds
from numpy import random
import pandas as pd
from sklearn import model_selection

# fold = 5
# df_label = pd.read_csv(
#     # 'result_1.csv'
#     r'D:\我的东西\实验室\老师工作安排\图像处理项目\imgs_30w.csv'
# )
# fold_lst = (len(df_label) // fold) * [i for i in range(fold)]
# r = random.random
# random.seed(10)
# random.shuffle(fold_lst)
# df_label['fold'] = fold_lst
# df_label.to_csv('imgs_30w_fold.csv', index=False)


if __name__ == "__main__":
    # Training data is in a csv file called train.csv
    # df = pd.read_csv("../txt/per_leafCid_250.csv")
    df = pd.read_csv("../../img1_0/img-300w/per_leafCid_250_1_2.csv")

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    # df = df.sample(frac=1).reset_index(drop=True)
    # fetch targets
    y = df.spuId.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the new csv with kfold column
    df.to_csv("../txt/per_leafCid_250_1_StratifiedKFold_2.csv", index=False)
