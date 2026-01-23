#!/bin/bash

#SBATCH --job-name=subtask1_evaluation
#SBATCH --output=../logs/eval%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

uv run python evaluation.py \
    --submission_path ../outputs/predictions.json \
    --key_path ../../data/dev/archehr-qa.xml \
    --quickumls_path quickumls/ \
    --out_file_path scores.json