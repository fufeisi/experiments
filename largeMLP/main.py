import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, argparse, time
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from quantize_activations.quant_layer import qLinear
from quantize_activations.test_grad import grad_diff

def train(args, model, device, train_loader, optimizer, epoch):
     model.train()
     loss_fun = torch.nn.CrossEntropyLoss()
     length = len(train_loader)
     start_time = time.time()
     for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          if batch_idx == 0:
               diff = grad_diff(model, loss_fun, data, target)
               print(f'Gradient SQNR {diff}.')
               output = model(data)
          else:
               output = model(data)
          # import pdb; pdb.set_trace()
          loss = loss_fun(output, target.view(-1))
          loss.backward()
          optimizer.step()
          if batch_idx % (length//args.log_nums) == 0:
               print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}. Epoch time:{:.2f}'.format(
                    epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                    100. * batch_idx / length, loss.item(), time.time()-start_time))


def test(model, device, test_loader):
     model.eval()
     correct = 0
     with torch.no_grad():
          for data, target in test_loader:
               data, target = data.to(device), target.to(device)
               output = model(data)
               pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
               correct += pred.eq(target.view_as(pred)).sum().item()
     print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
          correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
     return round(100. * correct / len(test_loader.dataset), 2)


class MLP(torch.nn.Module):
  def __init__(self, width, LinearLayer=None, act_fun=None, dtype=None, device='cuda'):
    super().__init__()
    self.LinearLayer = LinearLayer or torch.nn.Linear
    self.act_fun = act_fun or torch.relu
    self.dtype = dtype or torch.float32
    self.layers = torch.nn.ParameterList()
    iC = width[0]
    for oC in width[1:]:
      layer = self.LinearLayer(iC, oC, dtype=self.dtype, device=device)
      self.layers.append(layer)
      iC = oC
  
  def forward(self, x):
     x = x.view(-1, 28, 28)
     for layer in self.layers[:-1]:
          x = layer(x)
          x = self.act_fun(x)
     x = self.layers[-1](x)
     x = x.sum(dim=1)
     return x


def main():
     # Training settings
     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
     parser.add_argument('--batch-size', type=int, default=100)
     parser.add_argument('--epochs', type=int, default=10)
     parser.add_argument('--lr', type=float, default=0.001)
     parser.add_argument('--seed', type=int, default=0)
     parser.add_argument('--no-cuda', action='store_true', default=False,
                         help='disables CUDA training')
     parser.add_argument('--log_nums', type=int, default=10)
     parser.add_argument('--quan', type=int, default=0)
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
     dataset1 = datasets.MNIST('../data', train=True, transform=transforms.ToTensor())
     dataset2 = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

     model = MLP([28] + [64]*5 + [10], LinearLayer=qLinear).to(device)
     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

     test_acc = []
     training_time = 0
     for epoch in range(1, args.epochs + 1):
          start_time = time.time()
          train(args, model, device, train_loader, optimizer, epoch)
          training_time += time.time()-start_time
          test_acc.append(test(model, device, test_loader))

if __name__ == '__main__':
     main()
