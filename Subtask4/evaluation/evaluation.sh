#!/bin/bash
#SBATCH --job-name=subtask4_evaluation
#SBATCH --output=../logs/eval_subtask4_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

MODEL="anthropic-claude-sonnet-4-5_prompt_7"  # e.g. "anthropic-claude-sonnet-4-5_prompt_7"
DATASET="dev"

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Activate environment
source .venv/bin/activate

# Paths
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"
SUBMISSION_PATH="../outputs/${DATASET}/${MODEL}.json"
OUT_FILE_PATH="../results/${DATASET}/${MODEL}.json"

# Run scoring script
uv run python scoring_subtask_4.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[DONE] Subtask 4 scoring complete."
