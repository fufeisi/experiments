import re


def extract(contents, keywords, metrics):
     i, j, k = 0, 0, 0
     I, J = len(keywords), len(contents)
     res = {}
     curr_res = []
     while i < I:
          while j < J:
               if contents[j][:len(metrics[k])] == metrics[k]:
                    curr_res.append([float(contents[j].split(' ')[-3]), float(contents[j].split(' ')[-1][:-2])])
                    k = (k+1)%len(metrics)
               if i < I - 1 and contents[j][:len(keywords[i+1])] == keywords[i+1]:
                    if keywords[i] not in res:
                         res[keywords[i]] = []
                    res[keywords[i]].append(curr_res.copy())
                    curr_res = []
                    break
               j += 1
          i += 1
     if keywords[i-1] not in res:
          res[keywords[i-1]] = []
     res[keywords[i-1]].append(curr_res.copy())
     return res


if __name__ == '__main__':
     with open('slurm-78030.out') as f:
          lines = f.readlines()
          keywords = ['batch_size: 1024', 'batch_size: 2048']*5
          metrics = [str(i)+'/200:' for i in range(200)]
          res = extract(lines, keywords, metrics)
     avg = {}
     for key in res.keys():
          avg[key] = [[round(sum([res[key][j][i][0] for j in range(4)])/4, 4) for i in range(200)], 
          [round(sum([res[key][j][i][1] for j in range(4)])/4, 4) for i in range(200)]]
     print(avg)