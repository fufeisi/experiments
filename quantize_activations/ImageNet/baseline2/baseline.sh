#!/bin/bash

#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

srun python main.py --dist-url 'tcp://127.0.0.1:8325' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0