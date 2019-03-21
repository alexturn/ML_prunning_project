'''ResNet in PyTorch.
PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    
Masked ResNet50 for Fisher Pruning
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super (MaskedConv2d, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias), requires_grad=False)
        
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        
    def forward(self, input):
        bias = None
        if self.bias is not None:
            bias = self.bias * self.mask_bias
        
        weight = self.weight * self.mask_weight
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
    def truncate(self, value):
        if self.bias is not None:
            self.mask_bias.data = (self.bias_tmp > value).float()
        
        self.mask_weight.data = (self.weight_tmp > value).float()
        
    def erase_tmp(self):
        if self.bias is not None:
            self.bias_tmp = torch.zeros_like(self.bias.data)
        self.weight_tmp = torch.zeros_like(self.weight.data)


class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super (MaskedLinear, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.mask_bias = nn.Parameter(torch.ones_like(self.bias), requires_grad=False)
        
        self.mask_weight = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        
    def forward(self, input):
        bias = None
        if self.bias is not None:
            bias = self.bias * self.mask_bias
        
        weight = self.weight * self.mask_weight
        return F.linear(input, weight, bias)
    
    def truncate(self, value):
        if self.bias is not None: 
            self.mask_bias.data = (self.bias_tmp > value).float()
        self.mask_weight.data = (self.weight_tmp > value).float()
        
    def erase_tmp(self):
        if self.bias is not None:
            self.bias_tmp = torch.zeros_like(self.bias.data)
        self.weight_tmp = torch.zeros_like(self.weight.data)


def conv3x3(in_planes, out_planes, stride=1):
    return MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = MaskedConv2d(planes, self.expansion * planes, kernel_size=1,
                                  bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion * planes, kernel_size=1,
                             stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet50(num_classes=10):
    return ResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)
