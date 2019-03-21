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