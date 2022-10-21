from torch.profiler import profile, ProfilerActivity
from models import Model
from quantize_activations.quantization import qmatmul
import torch
from tool import sum_memory

device = 'cuda'
profiler_activity=[ProfilerActivity.CPU]
sort_by = 'cpu_memory_usage'
if device == 'cuda':
     profiler_activity.append(ProfilerActivity.CUDA)
     sort_by = 'cuda_memory_usage'

def main(has_sigmoid_functions, is_quantize):
     def id(x):
          return x
     if has_sigmoid_functions:
          act_fun = torch.sigmoid
     else:
          act_fun = id
     if is_quantize:
          matmul_op = qmatmul
     else:
          matmul_op = torch.mm
     x = torch.rand([1000, 2000], device=device)
     model = Model([2000]*11, act_fun=act_fun, matmul_op=matmul_op)  ## No activation functions!
     torch.cuda.reset_peak_memory_stats()
     with profile(activities=profiler_activity, profile_memory=True, record_shapes=True) as prof:
          loss = model(x).sum()
     print(prof.key_averages().table(sort_by=sort_by, row_limit=200))
     buffer_size = sum_memory(prof)
     print(f'Buffer Memory Usage:{buffer_size/1000**2} MB')
     print(f'Peak Memory: {torch.cuda.max_memory_allocated()/1000**2} MB')
     print(f'Peak Memory: {torch.cuda.max_memory_allocated()/1000**2} MB')
     loss.backward()
     for w in model.weights:
          _ = w.grad[0][0]


if __name__ == '__main__':
     main(False, True)
