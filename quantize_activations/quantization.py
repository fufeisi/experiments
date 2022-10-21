from turtle import forward
import torch
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


# class qLinear(torch.nn.Module):
#   def __init__(self) -> None:
#     super().__init__(i_dim, o_dim, bias=True, device='cuda', dtype=torch.float32)
#     self.w = torch.nn.parameter.Parameter(torch.randn(i_dim, o_dim, dtype=dtype, device=device))
#     self.b = torch.nn.parameter.Parameter(torch.randn(o_dim, dtype=dtype, device=device))
#     self.weights = torch.nn.ParameterList([self.w, self.b])
#   def forward(self, x):
#     x = self.qmatmul(x, self.w) + self.b