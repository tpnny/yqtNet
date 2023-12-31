import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import shutil

import yqtUtil.yqtDataset as yqtDataset
import yqtUtil.yqtEnvInfo as yqtEnvInfo
import yqtUtil.yqtNet as yqtNet
import yqtUtil.yqtRun as yqtRun

# 一些外部参数的设置
# batchsize
train_batchsize = 2
test_batchsize = 1
# 打印间隔
train_print_freq = 2
test_print_freq = 2

# 清除目录，每次重新保存模型和log
if os.path.exists("best.pth"):
    os.remove("best.pth")

if os.path.exists("logs_model"):
    shutil.rmtree("logs_model")
os.mkdir("logs_model")

yqtEnvInfo.printInfo()
device = yqtEnvInfo.yqtDevice()

# 加载数据
train_dataset = yqtDataset.yqtDataset(root_dir="/data/Data/agri", ntype="train")
test_dataset = yqtDataset.yqtDataset(root_dir="/data/Data/agri", ntype="test")

train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False)

print("train_dataset size:\t", train_dataset.size)
print("test_dataset size:\t", test_dataset.size)

# 网络初始化
model = yqtNet.yqtNet()

if os.path.exists("best.pth"):
    model.load_state_dict(torch.load("best.pth"))
model = model.to(device)

# 训练和测试 保存权重
optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

writer = SummaryWriter("logs_model")

# 此处注意是loss 还是prec loss是越小越好 prec是越大越好
best_prec = 10000
if os.path.exists("best.pth"):
    best_prec = yqtRun.test(model=model, dataLoader=test_loader, criterion=criterion, print_freq=test_print_freq,
                            epoch=0, device=device, writer=writer, type_="reg")

for epoch in range(0, 200):
    yqtRun.train(model=model, dataLoader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                 print_freq=train_print_freq, device=device, writer=writer, type_="reg")
    prec_ = yqtRun.test(model=model, dataLoader=test_loader, criterion=criterion, print_freq=test_print_freq,
                        epoch=epoch, device=device, writer=writer, type_="reg")

    if prec_ < best_prec:
        best_prec = prec_
        torch.save(model.state_dict(), "best.pth")

writer.close()
print('train end Best accuracy: ', best_prec)
