import torch, argparse
from no_quantized import main
from quantization import qConv2d_layer, qReLuLayer
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool import main_log
# Training settings
parser = argparse.ArgumentParser(description='CIFAR 10')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log_nums', type=int, default=10)
parser.add_argument('--milestones', default=[30*i for i in range(1, 4)])
parser.add_argument('--times', type=int, default=5)
args = parser.parse_args()
if args.lr == 0.0:
     args.lr = 0.05*(args.batch_size/512)**(1/2)
if __name__ == '__main__':
     res = []
     args_dict = vars(args)
     for item in args_dict:
          print(f'{item}: {args_dict[item]}')
     for i in range(args.times):
          best_acc, last_acc, training_time, run_epoch = main(i, Conv2dLayer=qConv2d_layer, act_fun=qReLuLayer, early_stop=88, args=args)
          res.append([best_acc, last_acc, training_time, run_epoch])
          if i == 0:
               peak_memo = torch.cuda.max_memory_allocated()/1000**2
               print(f'Peak Memory: {peak_memo} MB')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(4)]
     main_log('quantized_log.txt', __file__)
     for item in args_dict:
          main_log('quantized_log.txt', f'{item}: {args_dict[item]}')
     main_log('quantized_log.txt', str(avg_res))
