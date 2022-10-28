import torch 
from torchvision import datasets, transforms
from quant_layer import qLinear, qReLuLayer, qConv2d_layer, Conv2d_layer
from models import MLP, vgg16, vgg19_bn

def load_data(args):
     data_name = args.data_name
     assert data_name in ['MNIST', 'CIFAR10', 'CIFAR100']
     if data_name == 'MNIST':
          dataset1 = datasets.MNIST(args.data_path, train=True, transform=transforms.ToTensor())
          dataset2 = datasets.MNIST(args.data_path, train=False, transform=transforms.ToTensor())
          train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size)
          test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size)
     elif data_name == 'CIFAR10':
          normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
          train_loader = torch.utils.data.DataLoader(
               datasets.CIFAR10(root=args.data_path, train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
               ])),
               batch_size=args.batch_size, shuffle=True, pin_memory=True)

          test_loader = torch.utils.data.DataLoader(
               datasets.CIFAR10(root=args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
               ])),
               batch_size=args.batch_size, shuffle=False, pin_memory=True)
     elif data_name == 'CIFAR100':
          normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                             std=(0.2675, 0.2565, 0.2761))
          train_loader = torch.utils.data.DataLoader(
               datasets.CIFAR100(root=args.data_path, train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
               ])),
               batch_size=args.batch_size, shuffle=True, pin_memory=True)
          test_loader = torch.utils.data.DataLoader(
               datasets.CIFAR100(root=args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
               ])),
               batch_size=args.batch_size, shuffle=False, pin_memory=True)
     return train_loader, test_loader

def load_model(args):
     class_num = 10
     if args.data_name == 'CIFAR100':
          class_num = 100
     model_name = args.model_name
     assert model_name in ['MLP', 'vgg16', 'vgg19_bn']
     if args.quant:
          LinearLayer, act_fun, Conv2dLayer = qLinear, qReLuLayer, qConv2d_layer
     else:
          LinearLayer, act_fun, Conv2dLayer = torch.nn.Linear, torch.nn.ReLU, Conv2d_layer
     if model_name == 'MLP':
          return MLP([28*28, 512, 128, 10], LinearLayer=LinearLayer, act_fun=act_fun)
     elif model_name == 'vgg16':
          return vgg16(Conv2dLayer=Conv2dLayer, act_fun=act_fun)
     elif model_name == 'vgg19_bn':
          return vgg19_bn(Conv2dLayer=Conv2dLayer, act_fun=act_fun, class_num=class_num)
