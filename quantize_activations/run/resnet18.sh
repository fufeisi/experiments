#!/bin/bash

#SBATCH --time 72:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

echo "Job Started!"
srun python ../training.py --ffcv 0 --data_name IMAGENET --model_name resnet18 --quant 0 --num_workers 64 --batch_size 1024 --lr 0.5 --epochs 200 --early_stop 65 85 --times 2 --log_nums 100
