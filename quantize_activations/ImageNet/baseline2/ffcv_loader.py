import torch
import numpy as np
from typing import List
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256
def create_train_loader(train_dataset, this_device, num_workers, batch_size,
                         distributed, in_memory):
     print(f'Using {this_device}!! {torch.cuda.is_available()}')
     decoder = RandomResizedCropRGBImageDecoder((224, 224))
     image_pipeline: List[Operation] = [
          decoder,
          RandomHorizontalFlip(),
          ToTensor(),
          ToDevice(torch.device(this_device), non_blocking=True),
          ToTorchImage(),
          NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
     ]

     label_pipeline: List[Operation] = [
          IntDecoder(),
          ToTensor(),
          Squeeze(),
          ToDevice(torch.device(this_device), non_blocking=True)
     ]

     order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
     loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                         'image': image_pipeline,
                         'label': label_pipeline
                    },
                    distributed=distributed)

     return loader



def create_val_loader(val_dataset, this_device, num_workers, batch_size,
                         distributed):
     res_tuple = (256, 256)
     cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
     image_pipeline = [
          cropper,
          ToTensor(),
          ToDevice(torch.device(this_device), non_blocking=True),
          ToTorchImage(),
          NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
     ]

     label_pipeline = [
          IntDecoder(),
          ToTensor(),
          Squeeze(),
          ToDevice(torch.device(this_device),
          non_blocking=True)
     ]

     loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                         'image': image_pipeline,
                         'label': label_pipeline
                    },
                    distributed=distributed)
     return loader