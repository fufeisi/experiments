from turtle import forward
import torch
import math
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init

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
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

  def forward(self, input: Tensor) -> Tensor:
    return qmatmul(input, self.weight) + self.bias

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None
    )