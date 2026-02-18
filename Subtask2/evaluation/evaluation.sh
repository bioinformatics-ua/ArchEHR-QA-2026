#!/bin/bash
#SBATCH --job-name=subtask2_evaluation
#SBATCH --output=../logs/eval_subtask2_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Activate environment
source .venv/bin/activate

# Paths (edit as needed)
SUBMISSION_PATH="../outputs/dev/meta-llama-Llama-3-1-8B-Instruct_prompt_0.json"
KEY_PATH="../../data/dev/archehr-qa_key.json"
OUT_FILE_PATH="../results/dev/scores.json"

# Run scoring script
uv run python scoring_subtask_2.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[DONE] Subtask 2 scoring complete."
