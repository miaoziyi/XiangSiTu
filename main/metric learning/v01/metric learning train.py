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
parser.add_argument('-s', '--save_model_name', default=f'model/dim_{args.dimension}', required=True, help='训练模型保存名称')

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
# 'n_classes': 11973: 数据集中的类别数量，这里设置为11973。
# 'model_name': 'efficientnet_b3': 模型的名称，这里使用的是EfficientNet-B3模型。
# 'use_fc': True': 是否使用全连接层，这里使用了全连接层。
# 'fc_dim': weidu': 全连接层的维度大小，这里使用了一个变量weidu作为维度大小。
# 'dropout': 0.0': dropout概率，这里设置为0，即不使用dropout。
# 'loss_module': loss_module': 损失函数模块，这里使用了一个变量loss_module作为损失函数模块。
# 's': s': margin值，这里使用了一个变量s作为margin值。
# 'margin': 0.50': margin值，这里设置为0.5。
# 'ls_eps': 0.0': label-smoothing参数，这里设置为0，即不使用label-smoothing。
# 'theta_zero': 0.785': 额外的角度参数，这里设置为0.785。
# 'pretrained': True': 是否使用预训练模型，这里设置为True，即使用预训练模型。
# 在计算triplet loss时，margin是一个超参数，它控制了同一类样本和不同类样本之间的距离。对于同一类样本，我们希望它们的距离尽可能小，
# 而对于不同类样本，我们希望它们的距离尽可能大。因此，margin被设置为一个正数，
# 以确保同一类样本之间的距离小于不同类样本之间的距离。如果两个样本之间的距离小于margin，则不会对损失函数产生任何贡献。
# Triplet Loss的思想是通过最大化同类样本（即同一类别的样本）之间的相似度，最小化异类样本（即不同类别的样本）之间的相似度，来促进特征空间的有效分离。

# 具体来说，Triplet Loss的训练过程需要将一个样本分别与同类样本和异类样本组成的三元组作为输入。
# 对于一个样本，假设其特征向量为x，同类样本中与其距离最近的样本的特征向量为xp，异类样本中与其距离最近的样本的特征向量为xn，
# Triplet Loss的表达式可以表示为：
# 
# Label smoothing是指对于分类问题中的每一个样本，将其正确的标签设为1-ε，而将其他的标签都设为ε/(n_classes-1)，其中ε是一个比较小的数，n_classes是类别数。
# 这样做的目的是为了使模型对于正确标签的预测更加自信，而不是仅仅关注最大可能的标签。同时也可以减少模型过拟合训练数据的情况
# L = max(d(x, xp) - d(x, xn) + margin, 0)

# 其中，d(x, xp)表示样本x与同类样本xp之间的距离，d(x, xn)表示样本x与异类样本xn之间的距离，
# margin是一个预设的常数，表示同类样本与异类样本之间的距离应该具有的最小间隔。Triplet Loss的目标是最小化L，
# 使同类样本之间的距离尽可能小，异类样本之间的距离尽可能大，从而促进特征空间的有效分离。
# theta_zero 是在 ArcFace 损失函数中使用的一个超参数。
# ArcFace 是一种人脸识别算法中使用的损失函数，其可以将人脸图像嵌入到一个特征空间中，并通过计算嵌入向量之间的角度来度量它们之间的相似度。
# ArcFace 损失函数通过将一个人脸图像的嵌入向量移动到它所属的类别的球形中心来最大化该图像的分类间隔，同时最小化同一类别内的嵌入向量之间的角度。
在这个上下文中，参数's'是用来调节triplet loss中positive和negative样本之间距离的margin的，具体来说，margin的值为s乘以loss的标准差。当s的值越大时，margin也越大，positive和negative样本之间的距离就会越大，
这可以促进更好的类间分离，但同时也可能会导致类内距离变大。
相反，当s的值越小时，margin也越小，positive和negative样本之间的距离就会越小，这可以促进更好的类内紧密度，但同时也可能会导致类间距离变小。
因此，在使用triplet loss时，选择适当的s值是很重要的。
为什么用这些 还有哪些相似的替代
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# 使用Python内置的random库设置Python随机种子为seed。
# 将环境变量PYTHONHASHSEED的值设置为seed。这可以保证在使用哈希值的操作中也具有确定性。
# 使用NumPy库设置NumPy随机种子为seed。
# 使用PyTorch库设置PyTorch随机种子为seed。
# 使用PyTorch库设置CUDA设备上的随机种子为seed，以确保CUDA计算结果的一致性。
# 启用PyTorch中的deterministic模式，使得在使用CUDA时，每次计算的结果都相同。
# 综上所述，这段代码的作用是将多个随机数生成器的种子设置为同一个值，以保证在使用这些随机数生成器进行训练时，每次训练的结果都是一致的
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
如果SCHEDULER为ReduceLROnPlateau，则返回一个ReduceLROnPlateau学习率调度器，它在验证集损失不再下降时减少学习率。
如果SCHEDULER为CosineAnnealingLR，则返回一个CosineAnnealingLR学习率调度器，它按照余弦函数的形式逐渐降低学习率。
如果SCHEDULER为CosineAnnealingWarmRestarts，则返回一个CosineAnnealingWarmRestarts学习率调度器，
它在每个重启周期内按照余弦函数的形式逐渐降低学习率，重启周期的长度每次乘以T_mult。
其中，optimizer参数是一个优化器对象，用于管理模型参数的更新。函数的返回值是一个学习率调度器对象。

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
#             albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
#             albumentations.ShiftScaleRotate(
#              shift_limit=0.25, scale_limit=0.1, rotate_limit=0
#             ),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )
albumentations.Resize(DIM[0], DIM[1], always_apply=True)：调整图像大小，使其始终具有指定的高度和宽度。
albumentations.HorizontalFlip(p=0.5)：水平翻转图像，概率为0.5。
albumentations.VerticalFlip(p=0.5)：垂直翻转图像，概率为0.5。
albumentations.Rotate(limit=120, p=0.8)：在[-120, 120]度之间旋转图像，概率为0.8。
albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5)：随机调整图像的亮度，概率为0.5。
albumentations.Normalize()：对图像进行标准化，这里没有指定均值和标准差，因为在后面会使用预训练模型的均值和标准差。
ToTensorV2(p=1.0)：将图像转换为PyTorch张量。
这些方法是通过albumentations库实现的，可以用于图像增强和数据增广。
输入图像的亮度增加一个介于 0.09 到 0.6 之间的随机值
albumentations.Cutout 是一种数据增强技术，用于随机删除图像中的像素块，以增加训练集的多样性，避免过拟合。其中，参数 num_holes 指定要删除的像素块数，
max_h_size 和 max_w_size 分别指定删除的像素块的最大高度和宽度，fill_value 指定要用什么值来填充被删除的像素块， always_apply 指定是否应该始终应用该转换，p 是应用该转换的概率。

albumentations.ShiftScaleRotate 也是一种数据增强技术，包括平移、缩放和旋转
。其中，参数 shift_limit 指定允许平移的最大距离（作为图像高度和宽度的一部分），scale_limit 指定缩放因子的最大变化，
rotate_limit 指定旋转的最大角度。该转换也可以通过设置 always_apply 和 p 参数来控制其应用的概率。
为什么要注释掉？


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
# if not self._get_lr_called_within_step:：如果该方法不是在epoch步骤中被调用，则发出警告，建议使用 get_last_lr() 方法获取上一次计算的学习率。
# if self.last_epoch == 0:：如果上一个epoch为0，返回学习率列表为 lr_start 值的列表。
# lr = self._compute_lr_from_epoch(): 调用 _compute_lr_from_epoch() 方法计算当前epoch的学习率。
# self.last_epoch += 1: 更新 last_epoch 变量。
# return [lr for _ in self.optimizer.param_groups]: 返回优化器中所有参数组的学习率，即返回一个列表，列表长度为优化器参数组的个数，每个元素是一个学习率。
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
_get_closed_form_lr函数返回初始学习率，即在训练开始前的学习率；
_compute_lr_from_epoch函数用于计算当前学习率，它会根据当前epoch的值和lr_ramp_ep、lr_sus_ep、lr_decay等参数来计算当前学习率lr，
具体而言，当当前epoch小于lr_ramp_ep时，学习率会逐步增加到lr_max，当当前epoch在lr_ramp_ep和lr_ramp_ep+lr_sus_ep之间时，
学习率保持不变为lr_max，当当前epoch大于lr_ramp_ep+lr_sus_ep时，学习率会以指数方式逐步减小到lr_min。最终返回计算得到的当前学习率lr。

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
#           detach()和item()都是PyTorch中的tensor操作函数。

# detach()的作用是返回一个新的tensor，与原始tensor共享数据存储空间，
# 但是不会参与到计算图中，也就是不会有梯度回传。因此，detach()可以将计算结果从计算图中分离出来，用于避免一些梯度计算和优化问题。

# item()的作用是将一个标量张量转换为Python数值。它只能用于包含单个元素的张量，
# 例如tensor([2.0])，可以通过tensor([2.0]).item()将其转换为Python的浮点数2.0。这个函数主要用于将PyTorch张量的结果传递给其他Python库或进行打印输出等操作。
            tk0.set_postfix(Eval_Loss=loss_score.avg)

    return loss_score
？criterion
train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch)：定义了一个训练函数，包含以下参数：
dataloader：数据加载器，用于从数据集中读取数据并生成一个batch。
model：要训练的模型。
criterion：损失函数，用于计算模型输出与真实标签之间的差异。
optimizer：优化器，用于更新模型参数，使损失函数最小化。
device：指定使用的设备，如cuda或cpu。
scheduler：学习率调度器，用于动态调整学习率。
epoch：当前训练轮数。
model.train()：将模型设为训练模式，启用训练时特有的功能，如Dropout和Batch Normalization。
loss_score = AverageMeter()：定义一个AverageMeter对象，用于计算平均损失值。
tk0 = tqdm(enumerate(dataloader), total=len(dataloader))：创建一个进度条tqdm，用于显示训练进度。
for bi, d in tk0:：遍历数据加载器中的所有batch。
batch_size = d[0].shape[0]：获取当前batch中数据的数量。
images = d[0]：获取当前batch中的输入图像。
targets = d[1]：获取当前batch中的真实标签。
images = images.to(device)：将输入图像移动到指定的设备上。
targets = targets.to(device)：将真实标签移动到指定的设备上。
optimizer.zero_grad()：清除上一轮的梯度值，防止梯度累加。
output = model(images, targets)：使用模型对输入图像进行预测。
loss = criterion(output, targets)：计算预测值与真实标签之间的差异，并得到损失值。
loss.backward()：反向传播计算梯度。
optimizer.step()：根据计算出的梯度更新模型参数。
loss_score.update(loss.detach().item(), batch_size)：将当前batch的平均损失值更新到AverageMeter对象中。
tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])：更新进度条，显示当前训练轮数、平均损失值和当前学习率。
if scheduler is not None: scheduler.step()：如果使用了学习率调度器，则进行学习率调整。
return loss_score：返回本轮训练的平均损失值。

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
#     # 数据集对象
#     batch_size=TRAIN_BATCH_SIZE, # 每个小批量数据的样本数量
#     pin_memory=True,         # 是否将张量保存到 CUDA 固定内存中，加快数据传输
#     drop_last=True,          # 是否丢弃最后一个小批量数据，如果样本总数不能被 batch_size 整除
#     num_workers=NUM_WORKERS  # 加载数据时使用的子进程数量

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
            torch.save(model.state_dict(), f'model{args.save_model_name}.bin')
            print('best model found for epoch {}'.format(epoch))
            print(f'模型保存在model/{args.save_model_name}.bin')


run()
