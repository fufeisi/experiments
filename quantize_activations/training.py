import torch, time, argparse
from load_data_model import load_data, load_model
from trainer import train, test, test_topk
from tool import main_log
from tool import my_sqnr

# Training settings
parser = argparse.ArgumentParser(description='Activation Map Quantization')
parser.add_argument('--data_name', type=str, default='CIFAR10')
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--quant', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--log_nums', type=int, default=10)
parser.add_argument('--early_stop', type=int, nargs='+', default=[70, 90])
parser.add_argument('--milestones', default=None)
parser.add_argument('--times', type=int, default=5)
parser.add_argument('--sqnr', type=bool, default=False)
args = parser.parse_args() 
if args.lr == 0.0:
     from wiki import lr
     args.lr = lr[args.data_name]*args.batch_size
if args.milestones == None:
     args.milestones = [30*i for i in range(1, args.epochs//30+1)]

def main(seed, args):
     use_cuda = torch.cuda.is_available()
     torch.manual_seed(seed)
     if use_cuda:
          print('Running on cuda!!')
          device = torch.device("cuda")
     else:
          device = torch.device("cpu")
     
     train_loader, test_loader = load_data(args)
     model = load_model(args).to(device)
     optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
     if len(args.milestones) > 0:
          train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
     test_acc = [[], []]
     training_time = 0
     res = [0, 0, 0, 0, 0, args.epochs, args.epochs]
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          top1, top5 = test_topk(model, device, test_loader, [1, 5])
          test_acc[0].append(top1)
          test_acc[1].append(top5)
          if (res[-1] == args.epochs) and (top5 > args.early_stop[1]):
               res[-1] = epoch
          if (res[-2] == args.epochs) and (top1 > args.early_stop[0]):
               res[-2] = epoch
          if len(args.milestones) > 0:
               train_scheduler.step()
     res[:5] = max(test_acc[0]), max(test_acc[1]), test_acc[0][-1], test_acc[1][-1], training_time
     return res + my_sqnr.get_avg() if args.sqnr else res

if __name__ == '__main__':
     res = []
     args_dict = vars(args)
     for item in args_dict:
          print(f'{item}: {args_dict[item]}')
     for i in range(args.times):
          res.append(main(seed=i, args=args))
          if i == 0:
               peak_memo = torch.cuda.max_memory_allocated()/1000**2
               print(f'Peak Memory: {peak_memo} MB')
     avg_res = [round(sum([res[i][j] for i in range(len(res))])/len(res), 4) for j in range(len(res[0]))]
     log_name = args.data_name+'_log.txt'
     main_log(log_name, '\n')
     for item in args_dict:
          main_log(log_name, f'{item}: {args_dict[item]}')
     main_log(log_name, str(avg_res))
