from cmath import inf
import torch, os
import torch.distributed as dist

def sum_memory(prof):
     res = 0
     for item in prof.key_averages():
          if 'self_cuda_memory_usage' in dir(item):
               res +=item.self_cuda_memory_usage
     return res

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_log(file_name, content):
     with open(file_name, 'a') as f:
          f.write(content+'\n')

class QuantizeBuffer:
     def __init__(self):
          self.buffer_dict = {}
     def forward(self, x):
          if x not in self.buffer_dict:
               if len(self.buffer_dict) >= 1:
                    self.reset()
               # print('^'*100, x.dtype)
               self.buffer_dict[x] = torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
          return self.buffer_dict[x]
     def reset(self):
          self.buffer_dict = {}
          torch.cuda.empty_cache()

my_quantize = QuantizeBuffer()

class StoreSQNR:
     def __init__(self):
          self.sqnr = []
          self.layer = 0
          self.total_layers = 0
          self.freeze = False
     def save(self, x):
          if self.freeze:
               return 
          if self.layer == self.total_layers:
               self.total_layers += 1
               self.sqnr.append([])
          self.sqnr[self.layer].append(x)
          self.layer += 1
     def reset_layer(self):
          self.layer = 0
     def get(self):
          return self.sqnr
     def get_avg(self):
          return [sum(item)/len(item) for item in self.sqnr]

# class StoreSQNR:
#      def __init__(self):
#           self.sqnr = []
#      def reset(self):
#           self.sqnr = []
#      def save(self, x):
#           if x < inf:
#                self.sqnr.append(x)
#      def get_avg(self):
#           mean = sum(self.sqnr) / len(self.sqnr)
#           variance = sum([((x - mean) ** 2) for x in self.sqnr]) / len(self.sqnr)
#           std = variance ** 0.5
#           return mean, std

my_sqnr = StoreSQNR()
