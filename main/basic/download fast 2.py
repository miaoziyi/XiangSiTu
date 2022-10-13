from logging import exception
import threading
from time import sleep
import requests
from urllib import request
import os

# 表示进程结束的位置
end_list1 = -1
end_list2 = -1
end_list3 = -1
end_list4 = -1
end_list5 = -1
end_list6 = -1

f1 = '../txt/imgs_test_fast/1.txt'
f2 = '../txt/imgs_test_fast/2.txt'
f3 = '../txt/imgs_test_fast/3.txt'
f4 = '../txt/imgs_test_fast/4.txt'
f5 = '../txt/imgs_test_fast/5.txt'
f6 = '../txt/imgs_test_fast/6.txt'

# 保存处理异常跳过的链接
ff1 = open('skip1.txt', 'a+')
ff2 = open('skip2.txt', 'a+')
ff3 = open('skip3.txt', 'a+')
ff4 = open('skip4.txt', 'a+')
ff5 = open('skip5.txt', 'a+')
ff6 = open('skip6.txt', 'a+')


def get_imgs(fname, tname, ffname):
    # 默认的图片存储路径
    base_path = r'D:\xiangsitu\img_test_all/'

    # 读取txt文件
    with open(fname, 'r', encoding='utf8') as file:
        urls = file.readlines()
        # 计算链接地址条数
        n_urls = len(urls)
        n_404 = 0
        n_skip = 0
        ti = 1
        # 遍历链接地址下载图片
        for i, url in enumerate(urls):
            ti = ti + 1
            if ti % 100 == 0:
                ti = 1
                print('wait for next...')
                sleep(1)
            img_url = url.split('\t')[2].replace('\n', '')

            try:
                f = requests.get('http://img.17mjf.com/' + img_url, timeout=60)
                # 链接图片存在
                if f.status_code == 200:
                    path = base_path + '/'

                    name = img_url.strip().split('/')[-1]
                    imgPath = path + name
                    with open(imgPath, 'wb') as fp:
                        fp.write(f.content)

                    print('%i/%i' % (i + 1, n_urls), 'image', '进程' + tname)
                # 链接图片不存在
                else:
                    n_404 += 1
                    ffname.write(img_url + '\n')
                    print('%i/%i' % (i + 1, n_urls), 'no image', name)
            except:
                ffname.write(img_url + '\n')
                sleep(5)
                continue


# t1-t6对应txt1-6
t1 = threading.Thread(target=get_imgs, args=(f1, 't1', ff1))
t2 = threading.Thread(target=get_imgs, args=(f2, 't2', ff2))
t3 = threading.Thread(target=get_imgs, args=(f3, 't3', ff3))
t4 = threading.Thread(target=get_imgs, args=(f4, 't4', ff4))
t5 = threading.Thread(target=get_imgs, args=(f5, 't5', ff5))
t6 = threading.Thread(target=get_imgs, args=(f6, 't6', ff6))
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
