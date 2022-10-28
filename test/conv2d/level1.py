import torch, time
from conv2d import sconv2d

def main(conv2d_fun, device='cuda', seed=0):
     torch.manual_seed(seed)
     x = torch.rand([256, 64, 100, 100], device=device)
     w = torch.rand([3, 3, 64, 128], device=device)
     start_time = time.time()
     y = conv2d_fun(x, w)
     return time.time() - start_time


if __name__ == '__main__':
     for conv2d_fun in [torch.conv2d, sconv2d]:
          print(main(conv2d_fun))
