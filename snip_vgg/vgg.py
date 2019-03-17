import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


def get_mask_grad(module):
    grads = []
    if module.bias is not None:
        grads.append(module.mask_bias.grad.abs().view(-1))
    grads.append(module.mask_weight.grad.abs().view(-1))
    return grads


def truncate_mask(module, value):
    if module.mask_weight.grad is None:
        raise Exception('Compute gradients first')
    if module.bias is not None:
        module.mask_bias.data = (module.mask_bias.grad.abs() > value).float()
    module.mask_weight.data = (module.mask_weight.grad.abs() > value).float()
    module.freeze_mask()


def freeze_mask(module):
    if module.bias is not None:
        module.mask_bias.requires_grad = False
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


model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            MaskedLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            MaskedLinear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            MaskedLinear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
