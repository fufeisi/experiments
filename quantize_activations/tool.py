def sum_memory(prof):
     res = 0
     for item in prof.key_averages():
          if 'self_cuda_memory_usage' in dir(item):
               res +=item.self_cuda_memory_usage
     return res


def main_log(file_name, content):
     with open(file_name, 'a') as f:
          f.write(content+'\n')
