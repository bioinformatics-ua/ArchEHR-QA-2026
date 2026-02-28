#!/bin/bash
#SBATCH --job-name=subtask4_train
#SBATCH --output=../logs/train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "Activating environment..."
source ../inference/.venv/bin/activate

echo "Running training..."
python train.py