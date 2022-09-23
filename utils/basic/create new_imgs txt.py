import os

# 打开一个文件，可写模式
ff = open('../txt/new.txt', 'a')

for root, dirs_name, files_name in os.walk(r'D:\xiangsitu\img_test\online_result\online_result_06\in_images'):
    for i in files_name:
        ff.write('new/'+i+'\n')



# 提取imagePath列
# import pandas as pd

# image_ids = pd.read_csv(
#     r'D:\mzy\mzy\img1_0\train\data.csv'
# )
# image_ids2 = image_ids['imgPath']
# image_ids2.to_csv('img_test_path.csv', index=False)
