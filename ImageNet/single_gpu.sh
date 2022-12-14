#!/bin/bash

#SBATCH --time 72:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

srun python main.py --quan 1 --world-size 1 --rank 0 --workers 8 --batch-size 8192