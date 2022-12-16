#!/bin/bash

#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=feisi@meta.com
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=train

export TASK_NAME=cola
for QUAN in 0 1
do
     echo $QUAN
     python main.py --quan $QUAN --model_name_or_path roberta-large --task_name $TASK_NAME --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 2 --output_dir train/$TASK_NAME/$QUAN
done
echo 'bnb'
python main.py --optim adamw_bnb_8bit --model_name_or_path roberta-large --task_name $TASK_NAME --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 2 --output_dir train/$TASK_NAME/bnb