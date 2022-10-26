import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
from trainer import train, test
from models import vgg16

def main(seed, Conv2dLayer=None, act_fun=None, early_stop=100, args=None):
     # Training settings
     parser = argparse.ArgumentParser(description='CIFAR 10')
     parser.add_argument('--batch-size', type=int, default=1024)
     parser.add_argument('--epochs', type=int, default=100)
     parser.add_argument('--lr', type=float, default=0.05)
     parser.add_argument('--momentum', default=0.9, type=float)
     parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
     parser.add_argument('--seed', type=int, default=0)
     parser.add_argument('--no-cuda', action='store_true', default=False,
                         help='disables CUDA training')
     parser.add_argument('--log_nums', type=int, default=10)

     args = args or parser.parse_args([]) 
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

     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

     train_loader = torch.utils.data.DataLoader(
          datasets.CIFAR10(root='../../data', train=True, transform=transforms.Compose([
               transforms.RandomHorizontalFlip(),
               transforms.RandomCrop(32, 4),
               transforms.ToTensor(),
               normalize,
          ])),
          batch_size=args.batch_size, shuffle=True, pin_memory=True)

     test_loader = torch.utils.data.DataLoader(
          datasets.CIFAR10(root='../../data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               normalize,
          ])),
          batch_size=args.batch_size, shuffle=False, pin_memory=True)

     model = vgg16(Conv2dLayer=Conv2dLayer, act_fun=act_fun).to(device)
     optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

     test_acc = []
     training_time = 0
     res = [0, 0, 0, args.epochs]
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          test_acc.append(test(model, device, test_loader))
          if (res[-1] == args.epochs) and (test_acc[-1] > early_stop):
               res[-1] = epoch
               print('Reach early_stop!', res[-1])
     res[:3] = max(test_acc), test_acc[-1], round(training_time, 2)
     return res

if __name__ == '__main__':
     res = []
     with open('no_quantized_log.txt', 'a') as f:
          f.write('no quantized'+'\n')
     for i in range(10):
          best_acc, last_acc, training_time, run_epoch = main(i)
          res.append([best_acc, last_acc, training_time, run_epoch])
          with open('no_quantized_log.txt', 'a') as f:
               f.write(str(res[-1])+'\n')
               if i == 0:
                    peak_memo = torch.cuda.max_memory_allocated()/1000**2
                    print(f'Peak Memory: {peak_memo} MB')
                    f.write(f'Peak Memory: {peak_memo} MB'+'\n')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(4)]
     with open('no_quantized_log.txt', 'a') as f:
          f.write('Avg result: ' + str(avg_res)+'\n')