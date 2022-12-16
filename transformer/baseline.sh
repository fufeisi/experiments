#!/bin/bash

#SBATCH --time 168:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

python main.py
