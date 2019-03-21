
# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
sns.set(font_scale=1.2)

import copy

from masked_layers import MaskedConv2d
from vgg_module import VGG
from prune_tools import prune_vgg


torch.cuda.set_device(0)
cuda0 = torch.device('cuda:0')


def train_eval(vgg, optimizer, lr_scheduler=None):

    loss_track = 0.
    
    vgg.train()
    for input, target in tqdm(data_train_loader):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = vgg(input)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        loss_track += loss.item()

    vgg.eval()
    with torch.no_grad():
        
        correct, test_loss = 0., 0.
        for input, target in tqdm(data_test_loader):
            input, target = input.cuda(), target.cuda()
            output = vgg(input)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(data_test_loader.dataset)

        print('Average train loss: {:.4f}'.format(loss_track / len(data_train_loader)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_test_loader.dataset),
            100. * correct / len(data_test_loader.dataset)))
        if lr_scheduler:
            lr_scheduler.step()
    return 100. * correct / len(data_test_loader.dataset)


from torch.optim.lr_scheduler import LambdaLR, StepLR


def prune_and_learn(net, tresh):
    prune_vgg(net, N, trainloader_pruning, tresh)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                          lr=5e-4)
    n_epoch = 5
    accuracy_after_pruning = []
    for epoch in range(n_epoch):
        print('Epoch number {} starts....'.format(epoch))
        accuracy = train_eval(net, optimizer)
        accuracy_after_pruning.append(accuracy)
    return accuracy_after_pruning


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super(LinearLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr*np.minimum(-(self.last_epoch + 1)*1. /self.num_epochs + 1, 1.), 0 ))
        return res


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainset2 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)

trainloader_pruning = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)
data_train_loader = torch.utils.data.DataLoader(trainset2, batch_size=256,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
data_test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)



n = 3 #samples
accuracies = [{} for _ in range(n)]
tresholds = [15, 30, 45, 60, 75, 90]
N = 128
for i in range(n):
    vgg = VGG('VGG16', 100).to(cuda0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg.parameters()), 
                          lr=0.001)
    n_epoch = 1
    lr_scheduler = LinearLR(optimizer, n_epoch)
    for epoch in range(n_epoch):
        print('Epoch number {} starts....'.format(epoch))
        accuracy = train_eval(vgg, optimizer, lr_scheduler)
        if accuracies[i].get(0) is None:
            accuracies[i][0] = []
        accuracies[i][0].append(accuracy)
    #torch.save(vgg.state_dict(), 'vgg_cifar100_'+str(i)+'.pth')
    for j, tresh in enumerate(tresholds):
        accuracy_after_pruning = prune_and_learn(copy.deepcopy(vgg), tresh)
        accuracies[i][tresh] = accuracy_after_pruning


matrix = [[np.max(accuracies[i][t]) for t in [0]+tresholds] for i in range(n)]
print(matrix)
plt.plot(tresholds, np.mean(matrix, axis=0), alpha=0.5, c='blue')
for i in range(3):
    plt.scatter(tresholds, matrix[i], alpha=0.5, c='blue')
plt.savefig("cifar100.png")

