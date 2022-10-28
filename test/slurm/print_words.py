import argparse, torch, time
parser = argparse.ArgumentParser(description='Print')
parser.add_argument('--words', default='Wrong')
args = parser.parse_args()
def main(args):
     with open('log.txt', 'a') as f:
          f.write(f'Cuda:{torch.cuda.is_available()}', args.words, '/n')
     print(args.words)
     time.sleep(60)
if __name__ == '__main__':
     main(args)