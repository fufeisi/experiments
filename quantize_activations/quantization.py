import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import quantize_activations.tool as tl

quan = True

class qMatMul(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias):
    if quan and tl.my_quantize.train:
      qx = tl.my_quantize.forward(x)
      ctx.save_for_backward(qx, weight)
    else:
      ctx.save_for_backward(x, weight)
    with torch.no_grad():
      out = F.linear(x, weight, bias)  # (m, k) x (k, n)
    return out  # (m, n)
  
  @staticmethod
  def backward(ctx, dout):
    if quan:
      qx, weight = ctx.saved_tensors
      x = qx.dequantize()
    else:
      x, weight = ctx.saved_tensors
    dx, dweight = None, None
    if ctx.needs_input_grad[0]:
      dx = F.linear(dout, weight.T)
    if ctx.needs_input_grad[1]:
      dim = list(range(len(x.shape)-1))
      dweight = torch.tensordot(dout, x, dims=(dim, dim))
    return dx, dweight, dout

class qReLu(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    out = torch.relu(x)
    if quan and tl.my_quantize.train:
      qout = tl.my_quantize.forward(out)
      ctx.save_for_backward(qout)
    else:
      ctx.save_for_backward(out)
    return out
  
  @staticmethod
  def backward(ctx, dout):
    if quan:
      qout = ctx.saved_tensors[0]
      out = qout.dequantize()
    else:
      out = ctx.saved_tensors[0]
    dx = torch.mul(dout, torch.sign(out))
    return dx

class qConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    if quan and tl.my_quantize.train:
      qx = tl.my_quantize.forward(x)
      ctx.save_for_backward(qx, weight)
    else:
      ctx.save_for_backward(x, weight)
    ctx.bias_sizes_opt = 0 if bias is None else bias.shape[0]
    ctx.module = module
    ctx.stride = stride

    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups

    return F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

  @staticmethod
  def backward(ctx, grad_output):
    if quan:
      qx, weight = ctx.saved_tensors
      x = qx.dequantize()
    else:
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

class nqConv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    ctx.save_for_backward(x, weight)
    ctx.bias_sizes_opt = bias.shape[0] if bias else 0
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

# define a new max pooling layer using torch.autograd
class qMaxPool2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, kernel_size, stride):
    # apply the max pooling operation using a built-in function
    output = torch.nn.functional.max_pool2d(input, kernel_size, stride)

    # save the input and the parameters for the backward step
    if quan and tl.my_quantize.train:
      qinput = tl.my_quantize.forward(input)
      ctx.save_for_backward(qinput, kernel_size, stride)
    else:
      ctx.save_for_backward(input, kernel_size, stride)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    # retrieve the saved tensors
    if quan and tl.my_quantize.train:
      qinput, kernel_size, stride = ctx.saved_tensors
      input = qinput.dequantize()
    else:
      input, kernel_size, stride = ctx.saved_tensors

    # apply the max unpooling operation using a built-in function
    grad_input = torch.nn.functional.max_unpool2d(grad_output, input, kernel_size, stride)

    return grad_input, None, None

# define a new batch normalization layer using torch.autograd
class qBatchNorm2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    # compute the output of the batch normalization layer using built-in functions
    running_mean = torch.mean(input, dim=(0, 2, 3))
    running_var = torch.var(input, dim=(0, 2, 3), unbiased=False)
    weight = torch.ones(input.shape[1], 1, 1, device=input.device)
    bias = torch.zeros(input.shape[1], 1, 1, device=input.device)
    output = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias)

    # save the input and the computed parameters for the backward step
    if quan and tl.my_quantize.train:
      qinput = tl.my_quantize.forward(input)
      ctx.save_for_backward(qinput, running_mean, running_var, weight, bias)
    else:
      ctx.save_for_backward(input, running_mean, running_var, weight, bias)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    # retrieve the saved tensors
    if quan and tl.my_quantize.train:
      qinput, running_mean, running_var, weight, bias = ctx.saved_tensors
      input = qinput.dequantize()
    else:
      input, running_mean, running_var, weight, bias = ctx.saved_tensors

    # compute the gradient of the batch normalization layer using built-in functions
    grad_input = torch.nn.functional.batch_norm_backward(grad_output, input, running_mean, running_var, weight, bias, True)

    return grad_input
