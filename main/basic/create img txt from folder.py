import os

# 打开一个文件，可写模式
ff = open('new_test.txt', 'w')

for root, dirs_name, files_name in os.walk(r'D:\xiangsitu\img_test_all_error'):
    for i in files_name:
        ff.write(i+'\n')



# 提取imagePath列
# import pandas as pd

# image_ids = pd.read_csv(
#     r'D:\mzy\mzy\img1_0\train\data.csv'
# )
# image_ids2 = image_ids['imgPath']
# image_ids2.to_csv('img_test_path.csv', index=False)
