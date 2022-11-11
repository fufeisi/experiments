#!/bin/bash

#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

srun python main.py --ffcv 1 --world-size 8 --rank 0 --workers 64 --batch-size 1024