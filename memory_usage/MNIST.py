from torch.profiler import profile, record_function, ProfilerActivity
device = 'cuda'
profiler_activity=[ProfilerActivity.CPU]
sort_by = 'cpu_memory_usage'
if device == 'cuda':
     profiler_activity.append(ProfilerActivity.CUDA)
     sort_by = 'cuda_memory_usage'
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from hook_fun import register_max_memory 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def forward(model, device, optimizer):
    model.train()
    data, target = torch.rand([64, 1, 28, 28]), torch.randint(0, 9, [64])
    data, target = data.to(device), target.to(device)
    print('Forward Step: ')
    with profile(activities=profiler_activity, profile_memory=True, record_shapes=True) as prof:
        output = model(data)
        loss = F.nll_loss(output, target)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=200))

    print('Gradients: ')
    with profile(activities=profiler_activity, profile_memory=True, record_shapes=True) as prof:
        loss.backward()
    print(prof.key_averages().table(sort_by=sort_by, row_limit=200))

    print('Optimizer step: ')
    with profile(activities=profiler_activity, profile_memory=True, record_shapes=True) as prof:
        optimizer.step()
        optimizer.zero_grad()
    print(prof.key_averages().table(sort_by=sort_by, row_limit=200))

def main():
    print('Copy model from cpu to cuda')
    model = Net().to(device)
    summary(model, (1, 28, 28), 64)
    memory_usage = register_max_memory(model)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, 2):
        forward(model, device, optimizer)
    for name in memory_usage.outputs:
        print(f'{name}: {memory_usage.outputs[name]}')

if __name__ == '__main__':
    main()

