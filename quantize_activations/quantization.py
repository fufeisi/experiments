import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from tool import my_quantize, my_sqnr
from torch.ao.ns.fx.utils import compute_sqnr

id_quan = True
save_sqnr = False

class qMatMul(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, y):
    # print(x.storage().data_ptr())
    if id_quan:
      qx = my_quantize.forward(x)
    else:
      qx = torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
    ctx.save_for_backward(qx, y)
    if save_sqnr:
      my_sqnr.save(compute_sqnr(x, qx.dequantize()).item())
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

class qReLu(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    out = torch.relu(x)
    if id_quan:
      qout = my_quantize.forward(out)
    else:
      qout = torch.quantize_per_tensor_dynamic(out, torch.quint8, False)
    ctx.save_for_backward(qout)
    return out
  
  @staticmethod
  def backward(ctx, dout):
    qout = ctx.saved_tensors[0]
    out = qout.dequantize()
    dx = torch.mul(dout, torch.sign(out))
    return dx

class qConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    if id_quan:
      qx = my_quantize.forward(x)
    else:
      qx = torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
    if save_sqnr:
      my_sqnr.save(compute_sqnr(x, qx.dequantize()).item())
    ctx.bias_sizes_opt = bias.shape[0]
    ctx.save_for_backward(qx, weight)
    ctx.module = module
    ctx.stride = stride

    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups

    return F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

  @staticmethod
  def backward(ctx, grad_output):
    qx, weight= ctx.saved_tensors
    x = qx.dequantize()
    bias_sizes_opt = ctx.bias_sizes_opt
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups

    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(grad_output, x, weight, _pair(bias_sizes_opt),
                                               _pair(stride), _pair(padding), _pair(dilation),
                                               False, [0], groups, ctx.needs_input_grad[:3])

    return grad_input, grad_weight, grad_bias, None, None, None, None, None

class nqConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    ctx.save_for_backward(x, weight)
    ctx.bias_sizes_opt = bias.shape[0]
    ctx.module = module
    ctx.stride = stride

    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups
    out = F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return out

  @staticmethod
  def backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    bias_sizes_opt = ctx.bias_sizes_opt
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups
    
    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(grad_output, x, weight, _pair(bias_sizes_opt),
                                               _pair(stride), _pair(padding), _pair(dilation),
                                               False, [0], groups, ctx.needs_input_grad[:3])

    return grad_input, grad_weight, grad_bias, None, None, None, None, None
