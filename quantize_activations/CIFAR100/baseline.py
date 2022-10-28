'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, bias=True):
        super(VGG, self).__init__()
        print(f'Using Bias: {bias}.')
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512, bias=bias),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512, bias=bias),
            nn.ReLU(True),
            nn.Linear(512, 100, bias=bias),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if bias:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, bias=True):
     print(f'Using Bias: {bias}.')
     layers = []
     in_channels = 3
     for v in cfg:
          if v == 'M':
               layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          else:
               conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
               if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
               else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
               in_channels = v
     return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(bias):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], bias=bias), bias=bias)


def vgg11_bn(bias):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True, bias=bias), bias=bias)


def vgg13(bias):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], bias=bias), bias=bias)


def vgg13_bn(bias):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True, bias=bias), bias=bias)


def vgg16(bias):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], bias=bias), bias=bias)


def vgg16_bn(bias):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True, bias=bias), bias=bias)


def vgg19(bias):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], bias=bias), bias=bias)


def vgg19_bn(bias):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, bias=bias), bias=bias)


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
from trainer import train, test_topk


def main(bias):
     # Training settings
     parser = argparse.ArgumentParser(description='CIFAR100')
     parser.add_argument('--batch-size', type=int, default=512)
     parser.add_argument('--epochs', type=int, default=200)
     parser.add_argument('--lr', type=float, default=0.2)
     parser.add_argument('--momentum', default=0.9, type=float)
     parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
     parser.add_argument('--seed', type=int, default=0)
     parser.add_argument('--no-cuda', action='store_true', default=False,
                         help='disables CUDA training')
     parser.add_argument('--log_nums', type=int, default=10)
     parser.add_argument('--milestones', default=[60, 120, 160])
     args = parser.parse_args([])
     use_cuda = not args.no_cuda and torch.cuda.is_available()

     torch.manual_seed(args.seed)

     if use_cuda:
          print('Running on cuda!!')
          device = torch.device("cuda")
     else:
          device = torch.device("cpu")

     train_kwargs = {'batch_size': args.batch_size}
     test_kwargs = {'batch_size': args.batch_size}
     if use_cuda:
          cuda_kwargs = {'num_workers': 1,
                         'pin_memory': True,
                         'shuffle': True}
          train_kwargs.update(cuda_kwargs)
          test_kwargs.update(cuda_kwargs)

     # transform=transforms.Compose([
     #      transforms.ToTensor(),
     #      transforms.Normalize((0.1307,), (0.3081,))
     #      ])
     normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                        std=(0.2675, 0.2565, 0.2761))

     train_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100(root='../../data', train=True, transform=transforms.Compose([
               transforms.RandomHorizontalFlip(),
               transforms.RandomCrop(32, 4),
               transforms.ToTensor(),
               normalize,
          ])),
          batch_size=args.batch_size, shuffle=True, pin_memory=True)

     test_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100(root='../../data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               normalize,
          ])),
          batch_size=args.batch_size, shuffle=False, pin_memory=True)

     model = vgg19_bn(bias).to(device)
     optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
     train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)
     test_acc = []
     training_time = 0
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          test_acc.append(test_topk(model, device, test_loader, [1, 5]))
          train_scheduler.step()

if __name__ == '__main__':
     main(True)