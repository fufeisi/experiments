import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_layer import qLinear
from torch.nn import Linear
import torch, time, random
from torch.ao.ns.fx.utils import compute_sqnr

def main():
     for device in ['cuda', 'cpu']:
          for x_size_len in range(3, 6):
               size = [random.randint(20, 100), random.randint(20, 100)]
               qlayer, layer = qLinear(*size).to(device), Linear(*size).to(device)
               layer.weight = torch.nn.Parameter(qlayer.weight.clone().detach())
               layer.bias = torch.nn.Parameter(qlayer.bias.clone().detach())
               x_size = [random.randint(5, 10) for _ in range(x_size_len)] + [size[0]]
               x = torch.rand(x_size, device=device)*10 - 5
               x.requires_grad = True

               start_time = time.time()
               loss = torch.tanh(layer(x)).sum()
               loss.backward()
               noquan_time = time.time() - start_time
               x_grad = x.grad.clone().detach()
               
               for var in [x]:
                    var.grad = None

               start_time = time.time()
               qloss = torch.tanh(qlayer(x)).sum()
               qloss.backward()
               quan_time = time.time() - start_time

               # print(torch.norm(x_grad), torch.norm(x.grad), torch.norm(x_grad-x.grad))
               print(f'Forward Step sqnr: {compute_sqnr(torch.tanh(layer(x)), torch.tanh(qlayer(x))).item()}.')
               print(f'input sqnr: {compute_sqnr(x_grad, x.grad).item()}.')
               if size:
                    print(f'weight sqnr: {compute_sqnr(layer.weight.grad, qlayer.weight.grad).item()}.')
                    print(f'bias sqnr: {compute_sqnr(layer.bias.grad, qlayer.bias.grad).item()}.')
               # assert compute_sqnr(qx_grad, x.grad) >= 10
               # if size:
               #      print(compute_sqnr(qlayer.weight.grad, layer.weight.grad))
               #      assert compute_sqnr(qlayer.weight.grad, layer.weight.grad) >= 10
               print(f'Quan Time: {quan_time}; NonQuan Time: {noquan_time}')


if __name__ == '__main__':
     main()

