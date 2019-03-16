import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import masked_resnet

import numpy as np

from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from masked_layers import MaskedLinear

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

net = masked_resnet.resnet18()
net.fc = nn.Linear(512, 10, bias=True)
if torch.cuda.is_available(): net = net.cuda()

criterion = nn.CrossEntropyLoss()
sampler = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
for x, y in sampler: break
if torch.cuda.is_available():
    x, y = x.cuda(), y.cuda()
output = net(x)
loss = criterion(output, y)
loss.backward()

#compute percentile
agg_tensor = []
for child in masked_resnet.generator(net):
    agg_tensor += child.get_mask_grad()
agg_tensor = torch.cat(agg_tensor, dim=0).cpu().numpy()
value = np.percentile(agg_tensor, 90)

for child in masked_resnet.generator(net):
    child.truncate_mask(value)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                      lr=1e-2)
lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold=1e-3, 
                                 threshold_mode='abs', patience=25, factor=0.1, verbose=True)

n_epoch = 200
for epoch in range(n_epoch):
    print('Epoch number {} starts....'.format(epoch))
    loss_track = 0.
    
    net.train()
    for input, target in tqdm(trainloader):
        if torch.cuda.is_available(): 
            input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_track += loss.item()

    net.eval()
    with torch.no_grad():
        
        correct, test_loss = 0., 0.
        for input, target in tqdm(testloader):
            if torch.cuda.is_available(): 
                input, target = input.cuda(), target.cuda()
            output = net(input)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)

        print('Average train loss: {:.4f}'.format(loss_track / len(trainset)))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
        lr_scheduler.step(1. * correct / len(testloader.dataset))