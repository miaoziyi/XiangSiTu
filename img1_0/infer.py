import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader, Dataset  # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from PIL import Image
from tqdm import tqdm
from torchvision import models
from sklearn.model_selection import train_test_split


class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(dataset)  # some images are CMYK, Grayscale, check only RGB
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])
        classCategory = self.dataset[item][1]
        if self.transform:
            image = self.transform(image)
        return image, classCategory

    def checkChannel(self, dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")):  # Check Channels
                datasetRGB.append(dataset[index])
        return datasetRGB


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu or cpu
print(device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)
])  # train transform

# load pretrain model and modify...
# model = models.resnet50(pretrained=True)
model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load('D:/mzy/resnet50-19c8e357.pth'))

# If you want to do finetuning then set requires_grad = False
# Remove these two lines if you want to train entire model,
# and only want to load the pretrain weights.
for param in model.parameters():
    param.requires_grad = False  # 原本是False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.eval()

print("----> Loading checkpoint")
checkpoint = torch.load("./checpoint_epoch_4.pt")  # Try to load last checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

train = pd.read_csv(r'D:\xiangsitu\related\imgs_new.csv', keep_default_na=False)
train = train.loc[200001:400000]
train = train.reset_index(drop=True)
print(train)


# train[:5]

# train = train.reset_index(drop=True)

def RandomImagePrediction(filepath, i):
    # 图片存在
    if os.path.exists(filepath):
        try:
            img_array = Image.open(filepath).convert("RGB")
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
            img = data_transforms(img_array).unsqueeze(
                dim=0)  # Returns a new tensor with a dimension of size one inserted at the specified position.
            load = DataLoader(img)

            for x in load:
                x = x.to(device)
                pred = model(x)
                _, preds = torch.max(pred, 1)
                #         print(f"class : {preds}")
                if preds[0] == 1:
                    train1.loc[i, 'imgPath'] = filepath.split('/', 3)[3]
                else:
                    train1.loc[i, 'imgPath'] = ''
        except:
            train1.loc[i, 'imgPath'] = ''
    else:
        train1.loc[i, 'imgPath'] = ''


# 注意csv文件有时有列名，有时没有列名
train1 = train[['imgId', 'spuId', 'cid1', 'cid2', 'leafCid', 'img']]
print(len(train1))
print(train1[:5])

for i in range(len(train1)):
    if i % 1000 == 0:
        print(i)
    path = str(train1.loc[i]['cid1']) + '/' + str(train1.loc[i]['cid2']) + '/' + str(train1.loc[i]['leafCid']) + '/' + \
           train1.loc[i]['img'].split('/')[-1]
    RandomImagePrediction('D:/xiangsitu/imgs/' + path, i)

# print(train1[:15])
train2 = train1[train1['imgPath'] != '']
print(len(train2))
train3 = train1[train1['imgPath'] == '']
print(len(train3))

outputpath1 = 'img-10w/result_12.csv'
outputpath0 = 'img-10w/result_02.csv'
train2.to_csv(outputpath1, sep=',', index=False, header=True)
train3.to_csv(outputpath0, sep=',', index=False, header=True)
