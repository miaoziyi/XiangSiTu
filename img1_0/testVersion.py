# import torch
# import torchvision
#
# print(torchvision.__version__)
# print(torch.__version__)

# print('../input/imgs-1-cid2/imgs_1_cid2/11/1101/20220317201951828.jpg'.split('/',5)[5])
# from PIL import Image
#
# try:
#     img_array = Image.open('D:/xiangsitu/imgs/12/1201/120199/20220415044001285119940.jpg').convert("RGB")
# except:
#     print(0)

s = '92182,13990,12,1201,120101,images/crawler/20220318/20220318003038885.jpg,12/1201/120101/20220318003038885.jpg'
x = s.split(',')
print(x[2])