#!/bin/bash

#SBATCH --time 72:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

echo "Job Started!"
srun python ../training.py --data_name CIFAR10 --model_name vgg16 --quant 0 --batch_size 512 --num_workers 8 --epochs 200 --early_stop 90 100 --data_path ../../data --times 20
