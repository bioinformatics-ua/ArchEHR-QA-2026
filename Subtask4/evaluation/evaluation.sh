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
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
SUBTASK_DIR="${ROOT_DIR}/Subtask4"

source "${ROOT_DIR}/.venv/bin/activate"

# Paths
DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"
OUTPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"
LOGS_DIR="${SUBTASK_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

KEY_PATH="${DATA_DIR}/archehr-qa_key.json"
SUBMISSION_PATH="${OUTPUT_DIR}/${MODEL}.json"
OUT_FILE_PATH="${RESULTS_DIR}/${MODEL}.json"

# Run scoring script
uv run python scoring_subtask_4.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[DONE] Subtask 4 scoring complete."
