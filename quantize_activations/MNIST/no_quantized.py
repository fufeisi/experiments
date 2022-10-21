import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
from models import MLP
from trainer import train, test


def main(seed, LinearLayer=None, act_fun=None, early_stop=100):
     # Training settings
     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
     parser.add_argument('--batch-size', type=int, default=100)
     parser.add_argument('--epochs', type=int, default=20)
     parser.add_argument('--lr', type=float, default=0.001)
     parser.add_argument('--seed', type=int, default=seed)
     parser.add_argument('--no-cuda', action='store_true', default=False,
                         help='disables CUDA training')
     parser.add_argument('--log_nums', type=int, default=10)
     args = parser.parse_args([])
     use_cuda = not args.no_cuda and torch.cuda.is_available()

     torch.manual_seed(args.seed)

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

     # transform=transforms.Compose([
     #      transforms.ToTensor(),
     #      transforms.Normalize((0.1307,), (0.3081,))
     #      ])
     dataset1 = datasets.MNIST('../../data', train=True, transform=transforms.ToTensor())
     dataset2 = datasets.MNIST('../../data', train=False, transform=transforms.ToTensor())
     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

     model = MLP([28*28, 512, 128, 10], LinearLayer=LinearLayer, act_fun=act_fun)
     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

     test_acc = []
     training_time = 0
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          test_acc.append(test(model, device, test_loader))
          if test_acc[-1] > early_stop:
               return [max(test_acc), test_acc[-1], round(training_time, 2), epoch]
     return [max(test_acc), test_acc[-1], round(training_time, 2), args.epochs]

if __name__ == '__main__':
     res = []
     with open('log.txt', 'a') as f:
          f.write('no quantized'+'\n')
     for i in range(100):
          best_acc, last_acc, training_time, run_epoch = main(i, early_stop=98)
          res.append([best_acc, last_acc, training_time, run_epoch])
          with open('log.txt', 'a') as f:
               f.write(str(res[-1])+'\n')
               if i == 0:
                    peak_memo = torch.cuda.max_memory_allocated()/1000**2
                    print(f'Peak Memory: {peak_memo} MB')
                    f.write(f'Peak Memory: {peak_memo} MB'+'\n')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(4)]
     with open('log.txt', 'a') as f:
          f.write('Avg result: ' + str(avg_res)+'\n')
     