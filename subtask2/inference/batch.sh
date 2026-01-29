#!/bin/bash
#SBATCH --job-name=subtask2_llm_dev
#SBATCH --output=logs/subtask2_llm_dev_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -e

SPLIT=$1   # dev / test

echo "========================================"
echo "Subtask 2 – LLM labeling"
echo "Split: $SPLIT"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

uv run python inference.py \
    --split "$SPLIT" \
    --method llm \
    --output_dir results/

echo "[DONE] LLM labeling completed"
