import torch 
class FC(torch.nn.Module):
  def __init__(self, width, matmul_op=None, act_fun=None, dtype=None, device='cuda'):
    super().__init__()
    self.weights = torch.nn.ParameterList()
    self.dtype = dtype or torch.float32
    iC = width[0]
    for oC in width[1:]:
      w = torch.randn(iC, oC, dtype=self.dtype, device=device)
      self.weights.append(w)
      iC = oC

    self.matmul_op = matmul_op or torch.mm
    self.act_fun = act_fun or torch.relu
  
  def forward(self, x):
    x = x.view(-1, 28 * 28)
    for w in self.weights[:-1]:
      x = self.matmul_op(x, w)
      x = self.act_fun(x)
    x = self.matmul_op(x, self.weights[-1])
    return x
