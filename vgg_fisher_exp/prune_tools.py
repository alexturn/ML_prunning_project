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
from masked_layers import MaskedConv2d
from vgg_module import VGG

def erase_vgg_tmp(vgg):
    for child in vgg.features.children():
        if isinstance(child, MaskedConv2d):
            child.erase_tmp()

def prune_vgg(vgg, N, data_train_loader, tresh=90):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg.parameters()), 
                       lr=1e-3)
    erase_vgg_tmp(vgg)

    inc = 0
    for input, target in tqdm(data_train_loader):
        if inc > N - 1: break
        input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()
        output = vgg(input)
        loss = F.cross_entropy(output, target)
        loss.backward()
        for part in [vgg.features, vgg.classifier]:
            for child in part.children():

                if isinstance(child, MaskedConv2d):
                    child.weight_tmp = (child.weight_tmp + 
                        child.weight.data ** 2 * child.weight.grad.data ** 2 / N / 2.)

                    if child.bias is not None:
                        child.bias_tmp = (child.bias_tmp +
                          child.bias.data ** 2 * child.bias.grad.data ** 2 / N / 2.)
        inc += 1
    
    values = []
    for part in [vgg.features, vgg.classifier]:
        for child in part.children():
        
            if isinstance(child, MaskedConv2d):
                values += [child.weight_tmp.view(-1)]
            
                if child.bias is not None:
                    values += [child.bias_tmp.view(-1)]
                
    values = torch.cat(values, dim=0).cpu().detach().numpy()
    value = np.percentile(values, tresh)
    
    for part in [vgg.features, vgg.classifier]:
        for child in part.children():
            if isinstance(child, MaskedConv2d):
                child.truncate(value)