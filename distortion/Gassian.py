import torch 

def main(size):
     x = torch.randn(size)
     x = x - x.mean()
     zero_point, scale = 0, (x.max()-x.min())/255
     qx = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
     dqx = torch.dequantize(qx)
     error = torch.norm(x-dqx, p=2)/torch.norm(x, p=2)
     return error


if __name__ == '__main__':
     res = []
     for i in range(15, 25):
          res.append(main(256*2**i))
     print(res)