import torch.nn.functional as F
import torch

layer = torch.nn.Linear(5, 10)
x1 = torch.rand([32, 5])
x = torch.rand([2, 3, 5])
w = torch.rand([5, 10])
print(layer(x).shape)
print(F.linear(x, w.t()).shape)
