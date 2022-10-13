import os
import shutil
import glob

# 直接复制到一个文件夹里

# srcfile 需要复制、移动的文件
# dstpath 目的地址

def mycopyfile(srcfile, dstpath):  # 复制函数
    fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)  # 创建路径
    try:
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
    except:
        print(srcfile)


dst_dir = r'D:\xiangsitu\query_test/'  # 目的路径记得加斜杠
with open('../txt/query test/query-imgPath_new.txt', 'r', encoding='utf8') as file:
    urls = file.readlines()
    for i, url in enumerate(urls):
        print(i)
        # category = url.split(',')[2] + '/' + url.split(',')[3] + '/' + url.split(',')[4] + '/'
        # path = url.split(',')[6].replace('\n', '')
        # 针对 0 txt
        path = url.strip()
        # path = url.strip().split('/')[-1]
        mycopyfile(path, dst_dir)  # 复制文件
        # mycopyfile('D:/xiangsitu/imgs/'+path, dst_dir+category)  # 复制文件
