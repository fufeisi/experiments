import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from torch.nn.modules.utils import _pair
from quantization import qMatMul, qReLu, qConv2d, nqConv2d


def qlinear(x, y, z):
  return qMatMul.apply(x, y, z)

def qrelu(x):
  return qReLu.apply(x)

def qconv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
  return qConv2d.apply(x, weight, bias, stride, padding, dilation, groups, module)

def nqconv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
  return nqConv2d.apply(x, weight, bias, stride, padding, dilation, groups, module)

class qLinear(torch.nn.Linear):
  def forward(self, input: Tensor) -> Tensor:
      return qlinear(input, self.weight, self.bias)

class qReLuLayer(torch.nn.Module):
  def forward(self, input: Tensor) -> Tensor:
    return qrelu(input)

class qConv2d_layer(torch.nn.Conv2d):
  def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
      if self.padding_mode != 'zeros':
          return qconv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                          weight, bias, self.stride,
                          _pair(0), self.dilation, self.groups)
      return qconv2d(input, weight, bias, self.stride,
                      self.padding, self.dilation, self.groups)
  def forward(self, input: Tensor) -> Tensor:
    return self._conv_forward(input, self.weight, self.bias)

class Conv2d_layer(torch.nn.Conv2d):
  def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
      if self.padding_mode != 'zeros':
          return nqconv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                          weight, bias, self.stride,
                          _pair(0), self.dilation, self.groups)
      return nqconv2d(input, weight, bias, self.stride,
                      self.padding, self.dilation, self.groups)
  def forward(self, input: Tensor) -> Tensor:
    return self._conv_forward(input, self.weight, self.bias)

