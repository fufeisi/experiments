import torch, torchvision, numpy as np
from torchvision import datasets, transforms
from quant_layer import qLinear, qReLuLayer, qConv2d_layer, Conv2d_layer
from models import MLP, vgg16, vgg19_bn
from torch.utils.data.distributed import DistributedSampler


def load_data(args):
     data_name = args.data_name
     if data_name == 'MNIST':
          dataset1 = datasets.MNIST(args.data_path, train=True, transform=transforms.ToTensor())
          dataset2 = datasets.MNIST(args.data_path, train=False, transform=transforms.ToTensor())
          train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers)
          test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers)
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
               batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

          test_loader = torch.utils.data.DataLoader(
               datasets.CIFAR10(root=args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
               ])),
               batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
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
               batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
          test_loader = torch.utils.data.DataLoader(
               datasets.CIFAR100(root=args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
               ])),
               batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
     elif data_name == 'IMAGENET' and args.ffcv:
          IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
          IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
          from ffcv.loader import Loader, OrderOption
          from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, RandomHorizontalFlip
          from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
          res = (224, 224)
          decoder = RandomResizedCropRGBImageDecoder(res)

          # Data decoding and augmentation
          train_image_pipeline = [decoder, RandomHorizontalFlip(), ToTensor(), ToTorchImage(), ToDevice(args.local_rank), NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]
          label_pipeline = [IntDecoder(), ToTensor(), ToDevice(args.local_rank)]
          val_image_pipeline = [CenterCropRGBImageDecoder(res, 224/256), ToTensor(),ToDevice(args.local_rank, non_blocking=True),ToTorchImage(),NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]

          # Pipeline for each data field
          train_pipelines = {
          'image': train_image_pipeline,
          'label': label_pipeline
          }
          val_pipelines = {
          'image': val_image_pipeline,
          'label': label_pipeline
          }
          # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
          order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
          train_loader = Loader('/fsx/users/feisi/repos/experiments/ffcv-imagenet/train_500_0.50_90.ffcv', batch_size=args.batch_size, num_workers=args.num_workers,
                         order=order, pipelines=train_pipelines, distributed=args.distributed, seed=0)
          val_loader = Loader('/fsx/users/feisi/repos/experiments/ffcv-imagenet/val_500_0.50_90.ffcv', batch_size=args.batch_size, num_workers=args.num_workers,
                         order=OrderOption.SEQUENTIAL, pipelines=val_pipelines, distributed=args.distributed, seed=0)
          return train_loader, val_loader
     elif data_name == 'IMAGENET':
          import os
          from ImageNet.baseline.load_data import load_data as l_d
          data_path = "/datasets01_ontap/imagenet_full_size/061417"
          train_dir = os.path.join(data_path, "train")
          val_dir = os.path.join(data_path, "val")
          dataset_train, dataset_test = l_d(train_dir, val_dir)
          train_sampler = DistributedSampler(dataset_train, args.world_size, args.local_rank) if args.distributed else None
          test_sampler = DistributedSampler(dataset_test, args.world_size, args.local_rank) if args.distributed else None
          train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, sampler=train_sampler)
          test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, sampler=test_sampler)
     else:
          exit('No such data set~')
     return train_loader, test_loader

def load_model(args):
     model_name = args.model_name
     if args.quant:
          LinearLayer, act_fun, Conv2dLayer = qLinear, qReLuLayer, qConv2d_layer
     else:
          LinearLayer, act_fun, Conv2dLayer = torch.nn.Linear, torch.nn.ReLU, Conv2d_layer
     if model_name == 'MLP':
          return MLP([28*28, 512, 128, 10], LinearLayer=LinearLayer, act_fun=act_fun)
     elif model_name == 'vgg16':
          return vgg16(Conv2dLayer=Conv2dLayer, act_fun=act_fun)
     elif model_name == 'vgg19_bn':
          return vgg19_bn(Conv2dLayer=Conv2dLayer, act_fun=act_fun, class_num=args.class_num)
     elif model_name == 'resnet18':
          return torchvision.models.resnet18()

