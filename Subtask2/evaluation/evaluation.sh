#!/bin/bash
#SBATCH --job-name=subtask2_evaluation
#SBATCH --output=../logs/eval_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR"

source "${ROOT_DIR}/.venv/bin/activate"

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET="dev"               # dev / test / test-2026

# --- Evaluate a single file OR all files ---
# To score a single file, set SUBMISSION_FILE to the filename.
# To score all files in the output directory, leave it empty.
SUBMISSION_FILE=""          # e.g. "anthropic-claude-sonnet-4-5_prompt_7.json"

# --- Paths ---
SUBTASK_DIR="${ROOT_DIR}/Subtask2"
KEY_PATH="${SUBTASK_DIR}/data/${DATASET}/archehr-qa_key.json"
OUTPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"

mkdir -p "$RESULTS_DIR"

# =============================================================================
# SCORING
# =============================================================================

if [ -n "$SUBMISSION_FILE" ]; then
    # Score a single file
    echo "Scoring single file: ${SUBMISSION_FILE}"
    uv run python scoring_subtask_2.py \
        --submission_path "${OUTPUT_DIR}/${SUBMISSION_FILE}" \
        --key_path "$KEY_PATH" \
        --out_file_path "${RESULTS_DIR}/${SUBMISSION_FILE}"
else
    # Score ALL json files in the output directory
    echo "Scoring all files in ${OUTPUT_DIR}..."
    for f in "${OUTPUT_DIR}"/*.json; do
        if [ ! -f "$f" ]; then
            echo "  No JSON files found in ${OUTPUT_DIR}"
            break
        fi
        FNAME=$(basename "$f")
        echo "  Scoring: ${FNAME}"
        uv run python scoring_subtask_2.py \
            --submission_path "$f" \
            --key_path "$KEY_PATH" \
            --out_file_path "${RESULTS_DIR}/${FNAME}"
    done
fi

echo "[DONE] Evaluation complete."

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
echo "Aggregating results..."

uv run python aggregate_results.py --dataset "$DATASET"

echo "========================================"
