import sys

sys.path.append('../../../pytorch-image-models-master/')

from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from data import train_ImgDataset
from model import model as _ImgNet

# 将控制台的结果输出到a.log文件，可以改成a.txt
# sys.stdout = Logger('a.txt', sys.stdout)
# 导入库
import argparse

# 1. 定义命令行解析器对象
parser = argparse.ArgumentParser(description='Demo of argparse')

# 2. 添加命令行参数
parser.add_argument('--dimension', type=int, default=512)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('-t', '--train_csv_path', required=True, help='训练集文件路径')
args = parser.parse_args()
parser.add_argument('-s', '--save_model_name', default=f'dim_{args.dimension}', help='训练模型保存')

# 3. 从命令行中结构化解析参数
args = parser.parse_args()

# weidu = 512
# 修改
weidu = args.dimension
# 图片大小
DIM = (512, 512)

NUM_WORKERS = 0
TRAIN_BATCH_SIZE = args.batch
VALID_BATCH_SIZE = args.batch
# EPOCHS = 30
EPOCHS = 55
SEED = 42
# LR = 3e-4

print('cuda:', torch.cuda.is_available())

# torch.cuda.set_per_process_memory_fraction(0.3, 0)
device = torch.device('cuda')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"

################################################# MODEL ####################################################################

model_name = 'efficientnet_b3'  # efficientnet_b0-b7

################################################ Metric Loss and its params #######################################################
loss_module = 'arcface'  # 'cosface' #'adacos'
# 改 原 30
s = 15
# s = 30
# m = 0.5
# ls_eps = 0.0
# easy_margin = False

# 修改
name = args.save_model_name
# sys.stderr = Logger(f'log/{name}.log_file', sys.stderr)

####################################### Scheduler and its params ############################################################
# SCHEDULER = 'CosineAnnealingWarmRestarts' #'CosineAnnealingLR'
# factor=0.2 # ReduceLROnPlateau
# patience=4 # ReduceLROnPlateau
# eps=1e-6 # ReduceLROnPlateau
# T_max=10 # CosineAnnealingLR
# T_0=4 # CosineAnnealingWarmRestarts
# min_lr=1e-6


scheduler_params = {
    # 改
    # "lr_start": 1e-5,  # weidu：128，s:30的时候修改 效果不好或因s=30
    "lr_start": 6e-6,
    # "lr_start": 1e-5, # 最初
    "lr_max": 9e-6,
    # "lr_max": 3e-5,
    # "lr_max": 1e-5 * TRAIN_BATCH_SIZE,
    "lr_min": 1e-6,
    # 修改
    # "lr_ramp_ep": 55,
    "lr_ramp_ep": 20,  # s=10的时候修改 20 30
    "lr_sus_ep": 0,
    "lr_decay": 0.8,
}

############################################## Model Params ###############################################################
model_params = {
    'n_classes': 11973,  # 类别的数量，注意修改!!!!!!!!!!  145
    'model_name': 'efficientnet_b3',
    'use_fc': True,
    'fc_dim': weidu,
    'dropout': 0.0,
    'loss_module': loss_module,
    's': s,
    'margin': 0.50,
    'ls_eps': 0.0,
    'theta_zero': 0.785,
    'pretrained': True
}


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(SEED)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fetch_scheduler(optimizer):
    if SCHEDULER == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True,
                                      eps=eps)
    elif SCHEDULER == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
    elif SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
    return scheduler


def fetch_loss():
    loss = nn.CrossEntropyLoss()
    return loss


def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0], DIM[1], always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            # albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            # albumentations.ShiftScaleRotate(
            #  shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            # ),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0], DIM[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

class ImgScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                 lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8,
                 last_epoch=-1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(ImgScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]

        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) /
                  self.lr_ramp_ep * self.last_epoch +
                  self.lr_start)

        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max

        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay **
                  (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) +
                  self.lr_min)
        return lr


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0]
        targets = d[1]

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(images, targets)

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

    if scheduler is not None:
        scheduler.step()

    return loss_score


def eval_fn(data_loader, model, criterion, device):
    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            image = d[0]
            targets = d[1]

            image = image.to(device)
            targets = targets.to(device)

            output = model(image, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

    return loss_score


# 修改
# data = pd.read_csv('../../txt/per_leafCid_250_1_StratifiedKFold_5.csv')
data = pd.read_csv(args.train_csv_path)
data['filepath'] = data['imgPath'].apply(lambda x: os.path.join(r'D:\xiangsitu\imgs', x))
# print(data.head())

encoder = LabelEncoder()
# 类别
data['spuId'] = encoder.fit_transform(data['spuId'])


def run():
    train = data[data['kfold'] != 0].reset_index(drop=True)
    valid = data[data['kfold'] == 0].reset_index(drop=True)
    # Defining DataSet
    train_dataset = train_ImgDataset.ImgDataset(
        csv=train,
        transforms=get_train_transforms(),
    )

    valid_dataset = train_ImgDataset.ImgDataset(
        csv=valid,
        transforms=get_valid_transforms(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=NUM_WORKERS
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Device
    device = torch.device("cuda")

    # Defining Model for specific fold
    model = _ImgNet.ImgNet(**model_params)
    # print(model)
    # exit(1)
    model.to(device)

    # DEfining criterion
    criterion = fetch_loss()
    criterion.to(device)

    # Defining Optimizer with weight decay to params other than bias and layer norms
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #     optimizer_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    #             ]

    optimizer = torch.optim.Adam(model.parameters(), lr=scheduler_params['lr_start'])

    # Defining LR SCheduler
    scheduler = ImgScheduler(optimizer, **scheduler_params)

    # THE ENGINE LOOP
    best_loss = 10000
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)

        valid_loss = eval_fn(valid_loader, model, criterion, device)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('train_loss:', train_loss.avg, ' valid_loss:', best_loss)
            torch.save(model.state_dict(), f'{args.save_model_name}.bin')
            print('best model found for epoch {}'.format(epoch))


run()
