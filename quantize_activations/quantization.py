import torch
from torch.ao.quantization import HistogramObserver
from torch.nn.parameter import Parameter
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn import init
from torch import Tensor
from torch.utils.cpp_extension import load

# load the PyTorch extension
# cudnn_convolution = load(name="cudnn_convolution", sources=["/fsx/users/feisi/repos/experiments/quantize_activations/cudnn_convolution.cpp"], verbose=True)


class qMatMul(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, y):
    # s = 3.0 / 255.0
    # z = 0
    s = (x.max() - x.min()) / 255.
    z = (255 - x.max() / s).to(int)
    qx = torch.quantize_per_tensor(x, s, z, torch.quint8)
    # qy = torch.quantize_per_tensor(y, s, z, torch.quint8)
    ctx.save_for_backward(qx, y)
    with torch.no_grad():
      out = torch.mm(x, y)  # (m, k) x (k, n)
    return out  # (m, n)
  
  @staticmethod
  def backward(ctx, dout):
    qx, y = ctx.saved_tensors
    x = qx.dequantize()
    # y = qy.dequantize()
    dx = torch.mm(dout, y.T)
    dy = torch.mm(x.T, dout)
    return dx, dy

def qmatmul(x, y):
  return qMatMul.apply(x, y)

class qReLu(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    out = torch.relu(x)
    s = (out.max() - out.min()) / 255.
    z = 0
    qout = torch.quantize_per_tensor(out, s, z, torch.quint8)
    ctx.save_for_backward(qout)
    return out
  
  @staticmethod
  def backward(ctx, dout):
    qout = ctx.saved_tensors[0]
    out = qout.dequantize()
    dx = torch.mul(dout, torch.sign(out))
    return dx

def qrelu(x):
  return qReLu.apply(x)

class qLinear(torch.nn.Module):
  __constants__ = ['in_features', 'out_features']
  in_features: int
  out_features: int
  weight: Tensor

  def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device=None, dtype=None) -> None:
      factory_kwargs = {'device': device, 'dtype': dtype}
      super(qLinear, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
      if bias:
          self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
      else:
          self.register_parameter('bias', None)
      self.reset_parameters()

  def reset_parameters(self) -> None:
      # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
      # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
      # https://github.com/pytorch/pytorch/issues/57109
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
          fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
          bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
          init.uniform_(self.bias, -bound, bound)

  def forward(self, input: Tensor) -> Tensor:
      return qmatmul(input, self.weight.transpose(0, 1)) + self.bias

  def extra_repr(self) -> str:
      return 'in_features={}, out_features={}, bias={}'.format(
          self.in_features, self.out_features, self.bias is not None
      )

class qReLuLayer(torch.nn.Module):
  def forward(self, input: Tensor) -> Tensor:
    return qrelu(input)

class qConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    qx = torch.quantize_per_tensor_dynamic(x,torch.quint8, False)
    qweight = torch.quantize_per_tensor_dynamic(weight,torch.quint8, False)
    ctx.bias_sizes_opt = bias.shape[0]
    ctx.save_for_backward(qx, qweight)
    ctx.module = module
    ctx.stride = stride

    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups

    return F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

  @staticmethod
  def backward(ctx, grad_output):
    qx, qweight= ctx.saved_tensors
    x = qx.dequantize()
    weight = qweight.dequantize()
    bias_sizes_opt = ctx.bias_sizes_opt
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups

    # grad_input = grad_weight = grad_bias = None
    # if ctx.needs_input_grad[0]: grad_input = cudnn_convolution.convolution_backward_input(x.shape, weight, grad_output, stride, padding, dilation, groups)
    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(grad_output, x, weight, _pair(bias_sizes_opt),
                                               _pair(stride), _pair(padding), _pair(dilation),
                                               False, [0], groups, ctx.needs_input_grad[:3])

    return grad_input, grad_weight, grad_bias, None, None, None, None, None

def qconv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
  return qConv2d.apply(x, weight, bias, stride, padding, dilation, groups, module)

from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
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
    out = F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return out

  @staticmethod
  def backward(ctx, grad_output):
    x, weight, bias = ctx.saved_tensors
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups
    grad_input = grad_weight = grad_bias = None
    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(grad_output, x, weight, _pair(bias.shape[0]),
                                                _pair(stride), _pair(padding), _pair(dilation),
                                                False, [0], groups, ctx.needs_input_grad[:3])

    return grad_input, grad_weight, grad_bias, None, None, None, None, None

def sconv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
  return sConv2d.apply(x, weight, bias, stride, padding, dilation, groups, module)

