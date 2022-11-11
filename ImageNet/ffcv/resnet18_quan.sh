#!/bin/bash

#SBATCH --time 72:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

echo "Job Started!"
srun python train_imagenet.py --training.quan 1 --config-file ./configs/rn18_300_epochs.yaml --data.num_workers=64 --data.in_memory=1 --logging.folder=./resnet18/quan
