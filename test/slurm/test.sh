#!~/bin/sh
#SBATCH --time=2:00
srun -N1 -n1 python print_words.py --words test1

srun -N1 -n1 python print_words.py --words test2
