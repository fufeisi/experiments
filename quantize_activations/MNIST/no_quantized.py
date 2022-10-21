import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
from models import FC
from trainer import train, test


def main(seed, matmul_op=None, act_fun=None):
     # Training settings
     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
     parser.add_argument('--batch-size', type=int, default=100)
     parser.add_argument('--epochs', type=int, default=10)
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

     model = FC([28*28, 512, 128, 10], matmul_op=matmul_op, act_fun=act_fun)
     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
     print(f'Length of parameters: {len(list(model.parameters()))}')
     for w in model.parameters():
          print(w.size())

     test_acc = []
     training_time = 0
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          test_acc.append(test(model, device, test_loader))
     return [max(test_acc), test_acc[-1], round(training_time, 2)]

if __name__ == '__main__':
     res = []
     with open('log.txt', 'a') as f:
          f.write('no quantized'+'\n')
     for i in range(10):
          res.append(main(i))
          with open('log.txt', 'a') as f:
               f.write(str(res[-1])+'\n')
               if i == 0:
                    peak_memo = torch.cuda.max_memory_allocated()/1000**2
                    print(f'Peak Memory: {peak_memo} MB')
                    f.write(f'Peak Memory: {peak_memo} MB'+'\n')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(3)]
     with open('log.txt', 'a') as f:
          f.write('Avg result: ' + str(avg_res)+'\n')
     