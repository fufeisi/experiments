import torch, time
from tool import sum_memory, my_sqnr, my_quantize
from torch.profiler import profile, ProfilerActivity

def train(args, model, device, train_loader, optimizer, epoch):
     model.train()
     my_sqnr.freeze = False
     loss_fun = torch.nn.CrossEntropyLoss()
     length = len(train_loader)
     for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          my_quantize.reset()
          if batch_idx == 0:
               with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                    output = model(data)
                    my_quantize.reset()
               buffer_size = sum_memory(prof)
               print(f'Buffer Memory Usage:{buffer_size/1000**2} MB')
          else:
               output = model(data)
          loss = loss_fun(output, target)
          loss.backward()
          optimizer.step()
          if args.sqnr:
               my_sqnr.reset_layer()
          if batch_idx % (length//args.log_nums) == 0:
               print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
     my_sqnr.freeze = True


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


def test_topk(model, device, test_loader, topk=(1,)):
     model.eval()
     maxk = max(topk)
     res = [0 for _ in topk]
     with torch.no_grad():
          for data, target in test_loader:
               data, target = data.to(device), target.to(device)
               output = model(data)
               _, pred = output.topk(maxk, 1, True, True)
               pred = pred.t()
               correct = pred.eq(target.view(1, -1).expand_as(pred))
               for i, k in enumerate(topk):
                    res[i] += correct[:k].float().sum().item()
     print('\nTest set: Accuracy: {}/{} ({}%)\n'.format(
          res, len(test_loader.dataset),
          [round(100. * item / len(test_loader.dataset), 2) for item in res]))
     return [round(100. * item / len(test_loader.dataset), 2) for item in res]
