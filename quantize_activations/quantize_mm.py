import torch
class qMatMul(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, y):
    # s = 3.0 / 255.0
    z = 0
    s = (x.max() - x.min()) / 255.
    # z = (255 - x.max() / s).to(int)
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
