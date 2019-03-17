import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import vgg
import numpy as np
from logger import Logger
import time
import os
import yaml


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-(self.last_epoch + 1) * 1. / self.num_epochs + 1., 1.), 0.))
        return res


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cifar10')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--sparcity', default=95, type=int)
parser.add_argument('--log_dir', default='./logs/')
parser.add_argument('--vgg_type', default='vgg16')
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--prune_bs', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
os.makedirs(args.log_dir, exist_ok=True)

with open(os.path.join(args.log_dir, 'args.yml'), 'w') as f:
    yaml.dump(args, f, default_flow_style=False)

assert 0 <= args.sparcity <= 100

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fmt = {
    'time': '.3f'
}
logger = Logger('logs', base=args.log_dir, fmt=fmt)


transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='symmetric'),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.data == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
    N_CLASSES = 10
elif args.data == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='~/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True, transform=transform_test)
    N_CLASSES = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=3)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=3)
truncloader = torch.utils.data.DataLoader(trainset, batch_size=args.prune_bs, num_workers=3)

if args.vgg_type == 'vgg16':
    model = vgg.vgg16_bn(num_classes=N_CLASSES)
elif args.vgg_type == 'vgg19':
    model = vgg.vgg19_bn(num_classes=N_CLASSES)
model = model.to(device)

model.train()
x, y = map(lambda x: x.to(device), next(iter(trainloader)))
p = model(x)
loss = F.cross_entropy(p, y)
loss.backward()

agg_tensor = []
for child in model.modules():
    if isinstance(child, vgg.MaskedConv2d) or isinstance(child, vgg.MaskedLinear):
        agg_tensor += child.get_mask_grad()

agg_tensor = torch.cat(agg_tensor, dim=0).cpu().numpy()
value = np.percentile(agg_tensor, args.sparcity)

for child in model.modules():
    if isinstance(child, vgg.MaskedConv2d) or isinstance(child, vgg.MaskedLinear):
        child.truncate_mask(value)
torch.save(model.state_dict(), os.path.join(args.log_dir, 'model.torch'))

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
lr_scheduler = LinearLR(optimizer, args.epochs)

t0 = time.time()
for e in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        loss = F.cross_entropy(p, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    lr_scheduler.step()
    train_loss /= len(trainset)
    model.eval()
    with torch.no_grad():
        pred = torch.cat(list(map(lambda x: model(x[0].to(device)).argmax(dim=1), testloader))).cpu().numpy()
    test_acc = np.mean(pred == testset.targets)

    logger.add_scalar(e, 'loss', train_loss)
    logger.add_scalar(e, 'test.acc', test_acc)
    logger.add_scalar(e, 'time', time.time() - t0)
    t0 = time.time()
    logger.iter_info()
    logger.save()

    torch.save(model.state_dict(), os.path.join(args.log_dir, 'model.torch'))
