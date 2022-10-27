from numpy import array
import torch, argparse
from no_quantized import main
from quantization import Conv2d_layer
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool import main_log
# Training settings
parser = argparse.ArgumentParser(description='CIFAR 100')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log_nums', type=int, default=10)
parser.add_argument('--early_stop', default=[70, 90])
parser.add_argument('--milestones', default=[60, 120, 160])
args = parser.parse_args() 
if args.lr == 0.0:
     args.lr = 0.1*(args.batch_size/256)**(1/2)
if __name__ == '__main__':
     res = []
     args_dict = vars(args)
     for item in args_dict:
          print(f'{item}: {args_dict[item]}')
     for i in range(10):
          res.append(main(i, args=args, Conv2dLayer=Conv2d_layer))
          if i == 0:
               peak_memo = torch.cuda.max_memory_allocated()/1000**2
               print(f'Peak Memory: {peak_memo} MB')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(7)]
     main_log('cifar100_log.txt', __file__)
     for item in args_dict:
          main_log('cifar100_log.txt', f'{item}: {args_dict[item]}')
     main_log('cifar100_log.txt', str(avg_res))
