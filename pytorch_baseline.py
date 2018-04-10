#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 00:18:20 2018

@author: yk
"""

import torch
import os,sys
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets,transforms
from torch import optim
from myutils import AverageMeter,accuracy
from mynet import MobileNetV2
import tqdm,sys,os
best_acc=0
start_epoch=0
train_trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_data=datasets.CIFAR10('data',transform=train_trans,download=True)

train_dataloader=torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True,num_workers=3)

test_data=datasets.CIFAR10('data',train=False,transform=test_trans,download=True)

test_dataloader=torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=False,num_workers=3)

net=MobileNetV2(num_classes=10,in_c=3,scale=1).cuda()

torch.backends.cudnn.benchmark=True
LossFunc=CrossEntropyLoss().cuda()
lr=0.01
#optimizer=optim.Adam(net.parameters(),lr=lr,weight_decay=5e-4)
#optimizer=optim.RMSprop(net.parameters(),lr=lr, momentum=0.9,weight_decay=1e-4)
optimizer=optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=5e-4)
print('Loading Complete')
max_epoch=400
def calcu_acc():
    net.eval()
    count=0
    for batchstep,(image,label) in enumerate(test_dataloader):
        image=Variable(image).cuda()
        label=Variable(label).cuda()
        output=net(image)
        _,pre=torch.max(output.data,1)
        count+=pre.eq(label.data).cpu().sum()
    net.train()
    return count/len(test_data)*100

for epoch in range(start_epoch,max_epoch):
    lossMeter=AverageMeter()
    top1=AverageMeter()    
    with tqdm.tqdm(total=len(train_dataloader),leave=False) as pbar:
        for batchstep,(image,label) in enumerate(train_dataloader):
            image=Variable(image).cuda()
            label=Variable(label).cuda()
            output=net(image)
            loss=LossFunc(output,label)
            prec1=accuracy(output.cpu(), label.cpu())
            lossMeter.update(loss.data[0],image.size(0))
            top1.update(prec1[0].data[0],image.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f'{lossMeter.avg:.4f}',acc=f'{top1.avg:.3f}',lr=f'{lr:.1e}')
            pbar.update(1)
        acc=calcu_acc()
        print(f'\n epoch {epoch} acc:{acc:.4f} \n')
        lr*=0.93
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
