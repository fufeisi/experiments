import re


def extract(contents, keywords, metrics):
     i, j = 0, 0
     I, J = len(keywords), len(contents)
     res = {}
     curr_res = []
     while i < I:
          while j < J:
               if contents[j][:len(metrics[i])] == metrics[i]:
                    curr_res.append(float(''.join(c for c in contents[j] if c.isdigit() or c == '.')))
               if i < I - 1 and contents[j][:len(keywords[i+1])] == keywords[i+1]:
                    res[keywords[i]] = curr_res.copy()
                    curr_res = []
                    break
               j += 1
          i += 1
     res[keywords[i-1]] = curr_res.copy()
     return res


if __name__ == '__main__':
     with open('slurm-78796.out') as f:
          lines = f.readlines()
          keywords = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
          metrics = ['  eval_matthews_correlation']+['  eval_accuracy']*2 + ['  eval_combined_score'] + ['  eval_accuracy']*5
          res = extract(lines, keywords, metrics)
     print(res)