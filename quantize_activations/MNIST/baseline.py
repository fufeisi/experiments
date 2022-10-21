import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, bias=True):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        self.fc1 = nn.Linear(28 * 28, 512, bias=bias)
        self.fc2 = nn.Linear(512, 128, bias=bias)
        self.fc3 = nn.Linear(128, 10, bias=bias)
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
from trainer import train, test


def main(bias):
     # Training settings
     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
     parser.add_argument('--batch-size', type=int, default=100)
     parser.add_argument('--epochs', type=int, default=10)
     parser.add_argument('--lr', type=float, default=0.001)
     parser.add_argument('--seed', type=int, default=0)
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

     model = Net(bias).to(device)
     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
     print(f'Bias: {bias}. Length of parameters: {len(list(model.parameters()))}')
     for w in model.parameters():
          print(w.size())

     test_acc = []
     training_time = 0
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          test_acc.append(test(model, device, test_loader))

if __name__ == '__main__':
     main(False)