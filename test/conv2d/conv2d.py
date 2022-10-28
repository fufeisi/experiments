import torch
from torch.nn.modules.utils import _pair
from typing import Optional
from torch import Tensor
import torch.nn.functional as F

class Conv2d_layer(torch.nn.Conv2d):
  def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
      if self.padding_mode != 'zeros':
          return sconv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                          weight, bias, self.stride,
                          _pair(0), self.dilation, self.groups)
      return sconv2d(input, weight, bias, self.stride,
                      self.padding, self.dilation, self.groups)
  def forward(self, input: Tensor) -> Tensor:
    return self._conv_forward(input, self.weight, self.bias)

class sConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    ctx.save_for_backward(x, weight, bias)
    ctx.module = module
    ctx.stride = stride

    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups

    return F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

  @staticmethod
  def backward(ctx, grad_output):
    x, weight, bias = ctx.saved_tensors
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups
    grad_input = grad_weight = grad_bias = None

    if ctx.needs_input_grad[0]: grad_input = torch.nn.grad.conv2d_input(x.shape, weight, grad_output, stride, padding, dilation, groups)
    if ctx.needs_input_grad[1]:
        grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output, stride, padding, dilation, groups)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

    return grad_input, grad_weight, grad_bias, None, None, None, None, None

def sconv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
  return sConv2d.apply(x, weight, bias, stride, padding, dilation, groups, module)

