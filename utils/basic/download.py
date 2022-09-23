from time import sleep
import requests
from urllib import request
import os

# 默认的图片存储路径
base_path = 'D:/mzy/imgs/1/'

# 读取txt文件
with open('D:/mzy/mzy/img1_0/120207/result_1.txt', 'r', encoding='utf8') as file:
    urls = file.readlines()
    # 计算链接地址条数
    n_urls = len(urls)
    n_404 = 0
    ti=1
    # 遍历链接地址下载图片
    for i, url in enumerate(urls):
        ti=ti+1
        if ti%500==0:
            ti=1
            print('wait for next...')
            sleep(1)
        img_url = url.split(',')[-2]
        
        f = requests.get('http://img.17mjf.com/' + img_url, timeout=60)
        # 链接图片存在
        if f.status_code != 404:
            num = url.split(',')[0]
            category_1 = url.split(',')[3]
            category_2 = url.split(',')[4]
            category_3 = url.split(',')[5]
            path = base_path + category_1 + '/' + category_2 + '/' + category_3 + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            name = img_url.strip().split('/')[-1]
            imgPath = path + name
            with open(imgPath,'wb') as fp:
                fp.write(f.content)
            # 保存到当前目录的imgs文件夹下
            # request.urlretrieve('http://img.17mjf.com/' + url[img_url + 1:].replace('\n', ''),
            #                     path+name)
            # print('%i/%i' % (i + 1, n_urls), 'image', num)
        # 链接图片不存在
        else:
            n_404 += 1
            # print('%i/%i' % (i + 1, n_urls), 'no image', name)

    print('共', n_404, '张图片404无法下载')
