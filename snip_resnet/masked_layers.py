import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask_grad(module):
    grads = []
    if module.bias is not None: grads.append(module.mask_bias.grad.abs().view(-1))
    grads.append(module.mask_weight.grad.abs().view(-1))
    return grads

def truncate_mask(module, value):
    if module.mask_weight.grad is None: raise Exception('Compute gradients first')
    if module.bias is not None: 
        module.mask_bias.data = (module.mask_bias.grad.abs() > value).float()
    module.mask_weight.data = (module.mask_weight.grad.abs() > value).float()
    module.freeze_mask()

def freeze_mask(module):
    if module.bias is not None: module.mask_bias.requires_grad = False
    module.mask_weight.requires_grad = False


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        if self.bias is not None: 
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias))
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight))
        
    def forward(self, input):
        if self.bias is not None: 
            bias = self.bias * self.mask_bias
        else:
            bias = self.bias
        weight = self.weight * self.mask_weight
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def get_mask_grad(self):
        return get_mask_grad(self)

    def truncate_mask(self, value):
        truncate_mask(self, value)

    def freeze_mask(self):
        freeze_mask(self)


class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)
        if self.bias is not None: 
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias))
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight))
        
    def forward(self, input):
        if self.bias is not None: 
            bias = self.bias * self.mask_bias
        else:
            bias = self.bias
        weight = self.weight * self.mask_weight
        return F.linear(input, weight, bias)

    def get_mask_grad(self):
        return get_mask_grad(self)

    def truncate_mask(self, value):
        truncate_mask(self, value)

    def freeze_mask(self):
        freeze_mask(self)
