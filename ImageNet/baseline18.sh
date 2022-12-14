#!/bin/bash
#SBATCH --time 336:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train
for i in 1 2 3 4 5
do
     for batch in 1024 8192
     do
          python main.py --world-size 8 --rank 0 --workers 64 --batch-size $batch
     done
done