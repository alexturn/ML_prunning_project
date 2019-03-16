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
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias),
                                          requires_grad=False)
        
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight),
                                        requires_grad=False)
        
        
    def forward(self, input):
        if self.bias is not None: 
            bias = self.bias * self.mask_bias
        
        weight = self.weight * self.mask_weight
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)  
    
    def truncate(self, value):
        if self.bias is not None: 
            self.mask_bias.data = (self.bias_tmp > value).float()
        self.mask_weight.data = (self.weight_tmp > value).float()
        
    def erase_tmp(self):
        if self.bias is not None: self.bias_tmp = torch.zeros_like(self.bias.data)
        self.weight_tmp = torch.zeros_like(self.weight.data)


class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)
        if self.bias is not None: 
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias),
                                          requires_grad=False)
        
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight),
                                        requires_grad=False)
        
    def forward(self, input):
        if self.bias is not None: 
            bias = self.bias * self.mask_bias
        
        weight = self.weight * self.mask_weight
        return F.linear(input, weight, bias)
    
    
    def truncate(self, value):
        if self.bias is not None: 
            self.mask_bias.data = (self.bias_tmp > value).float()
        self.mask_weight.data = (self.weight_tmp > value).float()
        
    def erase_tmp(self):
        if self.bias is not None: self.bias_tmp = torch.zeros_like(self.bias.data)
        self.weight_tmp = torch.zeros_like(self.weight.data)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
        
    def forward(self, input):
        return input.view(input.size(0), -1)


def erase_lenet_tmp(LeNet5):
    for child in LeNet5.children():
        if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
            child.erase_tmp()


LeNet5 = nn.Sequential(
            MaskedConv2d(1, 6, kernel_size=(5, 5)),
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
            MaskedLinear(84, 10),
        )
LeNet5 = LeNet5.cuda()

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
data_train_loader_one = DataLoader(data_train, batch_size=1, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=1024)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, LeNet5.parameters()), 
                       lr=1e-3)

def prune_lenet(LeNet5, N):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, LeNet5.parameters()), 
                       lr=1e-3)
    erase_lenet_tmp(LeNet5)

    inc = 0
    for input, target in tqdm(data_train_loader_one):
        if inc > N - 1: break
        input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()
        output = LeNet5(input)
        loss = criterion(output, target)
        loss.backward()

        for child in LeNet5.children():
            
            if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
                child.weight_tmp = (child.weight_tmp + 
                    child.weight.data ** 2 * child.weight.grad.data ** 2 / len(data_train_loader_one) / 2.)
                
                if child.bias is not None:
                    child.bias_tmp = (child.bias_tmp +
                        child.bias.data ** 2 * child.bias.grad.data ** 2 / len(data_train_loader_one) / 2.)

        inc += 1
    
    values = []
    for child in LeNet5.children():
        
        if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
            values += [child.weight_tmp.view(-1)]
            
            if child.bias is not None:
                values += [child.bias_tmp.view(-1)]
                
    values = torch.cat(values, dim=0).cpu().detach().numpy()
    value = np.percentile(values, 90)
    
    for child in LeNet5.children():
        if isinstance(child, MaskedConv2d) or isinstance(child, MaskedLinear):
            child.truncate(value)

def train_eval(LeNet5, optimizer, lr_scheduler):

    loss_track = 0.
    
    LeNet5.train()
    for input, target in tqdm(data_train_loader):
        input, target = input.cuda(), target.cuda()
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
            input, target = input.cuda(), target.cuda()
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

if 'fisher_weights.pth' not in os.listdir():
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, LeNet5.parameters()), 
                          lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold=1e-3, 
                                     threshold_mode='abs', patience=3, factor=0.1, verbose=True)
    n_epoch = 15
    for epoch in range(n_epoch):
        print('Epoch number {} starts....'.format(epoch))
        train_eval(LeNet5, optimizer, lr_scheduler)
    torch.save(LeNet5.state_dict(), 'fisher_weights.pth')
else:
    LeNet5.load_state_dict(torch.load('fisher_weights.pth'))


prune_lenet(LeNet5, 60000)

LeNet5.eval()
with torch.no_grad():
    
    correct, test_loss = 0., 0.
    for input, target in tqdm(data_test_loader):
        input, target = input.cuda(), target.cuda()
        output = LeNet5(input)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_test_loader.dataset),
        100. * correct / len(data_test_loader.dataset)))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, LeNet5.parameters()), 
                      lr=1e-4)
lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold=1e-3, 
                                 threshold_mode='abs', patience=3, factor=0.1, verbose=True)
n_epoch = 10
for epoch in range(n_epoch):
    print('Epoch number {} starts....'.format(epoch))
    train_eval(LeNet5, optimizer, lr_scheduler)











