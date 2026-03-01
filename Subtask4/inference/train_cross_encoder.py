#!/bin/bash
#SBATCH --job-name=archEHR_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate inference

python train_cross_encoder.py