def main(name):
     res = [[], [], [], []]
     with open(name) as f:
          for line in list(f)[177:]:
               if line[0] == '[':
                    best_acc, acc, time, epoch = line[1:-2].split(',')
                    res[0].append(float(best_acc))
                    res[1].append(float(acc))
                    res[2].append(float(time))
                    res[3].append(int(epoch))
     avg_res = [sum(res[i])/len(res[i]) for i in range(4)]
     print(f'Name: {name}, avg:{avg_res}')

if __name__ == '__main__':
     for name in ['log.txt', 'quantized_log.txt']:
          main(name)