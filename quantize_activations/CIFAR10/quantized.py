import torch
from no_quantized import main
from quantization import qConv2d_layer, qReLuLayer


if __name__ == '__main__':
     res = []
     with open('quantized_log.txt', 'a') as f:
          f.write('quantized'+'\n')
     for i in range(10):
          best_acc, last_acc, training_time, run_epoch = main(i, Conv2dLayer=qConv2d_layer, act_fun=qReLuLayer)
          res.append([best_acc, last_acc, training_time, run_epoch])
          with open('quantized_log.txt', 'a') as f:
               f.write(str(res[-1])+'\n')
               if i == 0:
                    peak_memo = torch.cuda.max_memory_allocated()/1000**2
                    print(f'Peak Memory: {peak_memo} MB')
                    f.write(f'Peak Memory: {peak_memo} MB'+'\n')
     avg_res = [sum([res[i][j] for i in range(len(res))])/len(res) for j in range(4)]
     with open('quantized_log.txt', 'a') as f:
          f.write('Avg result: ' + str(avg_res)+'\n')
