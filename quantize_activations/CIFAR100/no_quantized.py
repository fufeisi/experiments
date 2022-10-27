import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
from trainer import train, test_topk
from models import vgg19_bn, vgg16
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
def main(seed, args, Conv2dLayer=None, act_fun=None):
     use_cuda = not args.no_cuda and torch.cuda.is_available()

     torch.manual_seed(seed)

     if use_cuda:
          print('Running on cuda!!')
          device = torch.device("cuda")
     else:
          device = torch.device("cpu")

     train_kwargs = {'batch_size': args.batch_size}
     test_kwargs = {'batch_size': args.batch_size}
     if use_cuda:
          cuda_kwargs = {'num_workers': 1,
                         'pin_memory': True,
                         'shuffle': True}
          train_kwargs.update(cuda_kwargs)
          test_kwargs.update(cuda_kwargs)

     normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                        std=(0.2675, 0.2565, 0.2761))
     train_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100(root='../../data', train=True, transform=transforms.Compose([
               transforms.RandomHorizontalFlip(),
               transforms.RandomCrop(32, 4),
               transforms.ToTensor(),
               normalize,
          ])),
          batch_size=args.batch_size, shuffle=True, pin_memory=True)

     test_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100(root='../../data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               normalize,
          ])),
          batch_size=args.batch_size, shuffle=False, pin_memory=True)

     model = vgg19_bn(Conv2dLayer=Conv2dLayer, act_fun=act_fun, class_num=100).to(device)
     optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
     train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)
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
          train_scheduler.step()
     res[:5] = max(test_acc[0]), max(test_acc[1]), test_acc[0][-1], test_acc[1][-1]. round(training_time, 2)
     return res

if __name__ == '__main__':
     res = []
     args_dict = vars(args)
     for item in args_dict:
          print(f'{item}: {args_dict[item]}')
     for i in range(10):
          res.append(main(i, args))
          if i == 0:
               peak_memo = torch.cuda.max_memory_allocated()/1000**2
               print(f'Peak Memory: {peak_memo} MB')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(7)]
     main_log('cifar100_log.txt', __file__)
     for item in args_dict:
          main_log('cifar100_log.txt', f'{item}: {args_dict[item]}')
     main_log('cifar100_log.txt', str(avg_res))