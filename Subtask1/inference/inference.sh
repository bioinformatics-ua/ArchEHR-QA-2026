#!/bin/bash

#SBATCH --job-name=subtask1_inference
#SBATCH --output=../logs/inference%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"


# --- Run the Inference Script ---
echo "Starting Python inference script..."
uv run python inference.py \
    --xml-file ../../data/dev/archehr-qa.xml \
    --prompt-file prompt.jsonl \
    --output-file ../outputs/predictions.json


# --- Run Evaluation Script ---
echo "Starting evaluation script..."
cd ../evaluation
uv run python evaluation.py \
    --submission_path ../outputs/predictions.json \
    --key_path ../../data/dev/archehr-qa.xml \
    --quickumls_path ../quickumls/ \
    --out_file_path ../results/predictions.json


echo "Evaluation complete. Job finished."
