import torch, argparse
from no_quantized import main
from quantization import qConv2d_layer, qReLuLayer
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool import main_log
# Training settings
parser = argparse.ArgumentParser(description='CIFAR 10')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log_nums', type=int, default=10)
args = parser.parse_args([])
if __name__ == '__main__':
     res = []
     for i in range(1):
          best_acc, last_acc, training_time, run_epoch = main(i, Conv2dLayer=qConv2d_layer, act_fun=qReLuLayer, args=args)
          res.append([best_acc, last_acc, training_time, run_epoch])
          if i == 0:
               peak_memo = torch.cuda.max_memory_allocated()/1000**2
               print(f'Peak Memory: {peak_memo} MB')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(4)]
     args_dict = vars(args)
     main_log('quantized_log.txt', __file__)
     for item in args_dict:
          main_log('quantized_log.txt', f'{item}: {args_dict[item]}')
     main_log('quantized_log.txt', str(avg_res))