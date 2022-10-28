import torch, math
# class FC(torch.nn.Module):
#   def __init__(self, width, matmul_op=None, act_fun=None, dtype=None, device='cuda'):
#     super().__init__()
#     self.weights = torch.nn.ParameterList()
#     self.dtype = dtype or torch.float32
#     iC = width[0]
#     for oC in width[1:]:
#       stdv = 1. / math.sqrt(iC)
#       w = torch.randn(iC, oC, dtype=self.dtype, device=device)
#       w.data.uniform_(-stdv, stdv)
#       self.weights.append(w)
#       iC = oC

#     self.matmul_op = matmul_op or torch.mm
#     self.act_fun = act_fun or torch.relu
  
#   def forward(self, x):
#     x = x.view(-1, 28 * 28)
#     for w in self.weights[:-1]:
#       x = self.matmul_op(x, w)
#       x = self.act_fun(x)
#     x = self.matmul_op(x, self.weights[-1])
#     return x


class MLP(torch.nn.Module):
  def __init__(self, width, LinearLayer=None, act_fun=None, dtype=None, device='cuda'):
    super().__init__()
    self.LinearLayer = LinearLayer or torch.nn.Linear
    self.act_fun = act_fun or torch.relu
    self.dtype = dtype or torch.float32
    self.layers = torch.nn.ParameterList()
    iC = width[0]
    for oC in width[1:]:
      layer = self.LinearLayer(iC, oC, dtype=self.dtype, device=device)
      self.layers.append(layer)
      iC = oC
  
  def forward(self, x):
    x = x.view(-1, 28 * 28)
    for layer in self.layers[:-1]:
      x = layer(x)
      x = self.act_fun(x)
    x = self.layers[-1](x)
    return x