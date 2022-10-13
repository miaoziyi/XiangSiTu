from time import sleep
import requests
from urllib import request
import os

# 默认的图片存储路径
base_path = r'D:\xiangsitu\img_test_all/'
ff = open('../txt/error/download_test_error.txt', 'a')

# 读取txt文件
with open(r'D:\xiangsitu\related/imgs_test.txt', 'r', encoding='utf8') as file:
    urls = file.readlines()
    # 计算链接地址条数
    n_urls = len(urls)
    print('共{}条数据'.format(n_urls))
    ti = 0
    # 遍历链接地址下载图片
    for i, url in enumerate(urls):
        ti = ti + 1
        if ti % 500 == 0:
            ti = 0
            print('wait for next...')
            sleep(1)
        img_url = url.split('\t')[2].replace('\n', '')

        f = requests.get('http://img.17mjf.com/' + img_url, timeout=60)
        # 链接图片存在
        if f.status_code == 200:
            path = base_path + '/'

            name = img_url.strip().split('/')[-1]
            imgPath = path + name
            with open(imgPath, 'wb') as fp:
                fp.write(f.content)
        # 链接图片存在问题
        else:
            ff.write(img_url + '\n')
