import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import random
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchmodels

import torch.nn.functional as F

from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        if self.bias is not None: 
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias))
        
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight))
        
    def forward(self, input):
        if self.bias is not None: 
            bias = self.bias * self.mask_bias
        
        weight = self.weight * self.mask_weight
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def get_mask_grad(self):
        grads = []
        if self.bias is not None: grads.append(self.mask_bias.grad.abs().view(-1))
        grads.append(self.mask_weight.grad.abs().view(-1))
        return grads
    
    def truncate(self, value):
        if self.mask_weight.grad is None: 
            print('Compute grad first!')
            return
        if self.bias is not None: 
            self.mask_bias.data = (self.mask_bias.grad.abs() > value).float()
        self.mask_weight.data = (self.mask_weight.grad.abs() > value).float()
        self.freeze_mask()
    
    def freeze_mask(self):
        if self.bias is not None: self.mask_bias.requires_grad = False
        self.mask_weight.requires_grad = False


class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)
        if self.bias is not None: 
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias))
        
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight))
        
    def forward(self, input):
        if self.bias is not None: 
            bias = self.bias * self.mask_bias
        
        weight = self.weight * self.mask_weight
        return F.linear(input, weight, bias)
    
    def get_mask_grad(self):
        masks = []
        if self.bias is not None: masks.append(self.mask_bias.grad.abs().view(-1))
        masks.append(self.mask_weight.grad.abs().view(-1))
        return masks
    
    def truncate(self, value):
        if self.mask_weight.grad is None: 
            print('Compute grad first!')
            return
        if self.bias is not None: 
            self.mask_bias.data = (self.mask_bias.grad.abs() > value).float()
        self.mask_weight.data = (self.mask_weight.grad.abs() > value).float()
        self.freeze_mask()

    def freeze_mask(self):
        if self.bias is not None: self.mask_bias.requires_grad = False
        self.mask_weight.requires_grad = False


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
        
    def forward(self, input):
        return input.view(input.size(0), -1)

LeNet5 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            MaskedConv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            MaskedConv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=1024)

criterion = nn.CrossEntropyLoss()

sampler = DataLoader(data_train, batch_size=100, shuffle=True)
for x, y in sampler: break
output = LeNet5(x)
loss = criterion(output, y)
loss.backward()

#compute percentile
agg_tensor = []
for child in LeNet5.children():
    if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
        agg_tensor += child.get_mask_grad()
agg_tensor = torch.cat(agg_tensor, dim=0).numpy()
value = np.percentile(agg_tensor, 95)

for child in LeNet5.children():
    if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
        child.truncate(value)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, LeNet5.parameters()), 
                      lr=1e-3)
lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold=1e-3, 
                                 threshold_mode='abs', patience=3, factor=0.1, verbose=True)

n_epoch = 30
for epoch in range(n_epoch):
    print('Epoch number {} starts....'.format(epoch))
    loss_track = 0.
    
    LeNet5.train()
    for input, target in tqdm(data_train_loader):

        optimizer.zero_grad()
        output = LeNet5(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_track += loss.item()

    LeNet5.eval()
    with torch.no_grad():
        
        correct, test_loss = 0., 0.
        for input, target in tqdm(data_test_loader):
        
            output = LeNet5(input)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_test_loader.dataset)

        print('Average train loss: {:.4f}'.format(loss_track / len(data_train)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_test_loader.dataset),
            100. * correct / len(data_test_loader.dataset)))
        lr_scheduler.step(1. * correct / len(data_test_loader.dataset))

    # num, num_non = 0., 0.
    # for child in LeNet5.children():

    #     if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
    #         num += (child.mask_weight >= 0.).float().sum()
    #         num_non += (child.mask_weight < 1.).float().sum() 

    # print(num_non, num, num_non / num)













