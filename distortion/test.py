import torch

x = torch.rand(10)
dx = torch.dequantize(x)
print(x)
print(dx)
