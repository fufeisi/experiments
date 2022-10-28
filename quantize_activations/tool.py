from cmath import inf
import torch

def sum_memory(prof):
     res = 0
     for item in prof.key_averages():
          if 'self_cuda_memory_usage' in dir(item):
               res +=item.self_cuda_memory_usage
     return res


def main_log(file_name, content):
     with open(file_name, 'a') as f:
          f.write(content+'\n')

class QuantizeBuffer:
     def __init__(self):
          self.buffer_dict = {}
     def forward(self, x):
          if id(x) not in self.buffer_dict:
               self.buffer_dict[id(x)] = torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
          return self.buffer_dict[id(x)]
     def reset(self):
          self.buffer_dict = {}
          torch.cuda.empty_cache()

my_quantize = QuantizeBuffer()

# class StoreSQNR:
#      def __init__(self):
#           self.sqnr = []
#           self.layer = 0
#           self.total_layers = 0
#      def reset(self, total_layers):
#           self.sqnr = []
#           self.layer = 0
#           self.total_layers = 0
#      def save(self, x):
#           if self.layer == self.total_layers:
#                self.total_layers += 1
#                self.sqnr.append([])
#           self.sqnr[self.layer].append(x)
#           self.layer += 1
#      def reset_layer(self):
#           self.layer = 0
#      def get(self):
#           return self.sqnr
#      def get_avg(self):
#           return [sum(item)/len(item) for item in self.sqnr]

class StoreSQNR:
     def __init__(self):
          self.sqnr = []
     def reset(self):
          self.sqnr = []
     def save(self, x):
          if x < inf:
               self.sqnr.append(x)
     def get_avg(self):
          mean = sum(self.sqnr) / len(self.sqnr)
          variance = sum([((x - mean) ** 2) for x in self.sqnr]) / len(self.sqnr)
          std = variance ** 0.5
          return mean, std

my_sqnr = StoreSQNR()
