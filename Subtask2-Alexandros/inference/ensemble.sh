#!/bin/bash
#SBATCH --job-name=subtask2_ensemble
#SBATCH --output=../logs/ensemble_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET="dev"                 # dev / test / test-2026
JUDGE_MODEL="openai/gpt-4.1"
MODE="select"                # select = pick best candidate; merge = synthesise
TEMPERATURE=0.0
MAX_TOKENS=4096
FALLBACK=0                   # 0-based index of fallback candidate file

# --- Candidate files to ensemble (best runs) ---
DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"

CANDIDATES=(
    "${OUTPUT_DIR}/anthropic-claude-sonnet-4-5_prompt_7.json"
    "${OUTPUT_DIR}/anthropic-claude-opus-4-6_prompt_7.json"
    "${OUTPUT_DIR}/google-gemini-2-5-flash_prompt_7.json"
)

# --- Output filename ---
OUTPUT_FILE="ensemble_${MODE}_$(echo $JUDGE_MODEL | tr '/' '-' | tr '.' '-')_top${#CANDIDATES[@]}.json"

# Load .env for API keys
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# =============================================================================
# ENSEMBLE
# =============================================================================
echo "[1/2] Running LLM-as-Judge ensemble (${MODE} mode)..."
echo "  Judge:      ${JUDGE_MODEL}"
echo "  Candidates: ${#CANDIDATES[@]} files"
echo "  Dataset:    ${DATASET}"

uv run python ensemble.py \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --candidate-files "${CANDIDATES[@]}" \
    --judge-model "$JUDGE_MODEL" \
    --mode "$MODE" \
    --output-file "${OUTPUT_DIR}/${OUTPUT_FILE}" \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --fallback $FALLBACK

echo "[1/2] Ensemble complete."

# =============================================================================
# EVALUATION
# =============================================================================
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"

if [ ! -f "${KEY_PATH}" ]; then
    echo "[2/2] No key file found for ${DATASET}, skipping evaluation."
    echo "========================================"
    echo "Output: ${OUTPUT_DIR}/${OUTPUT_FILE}"
    exit 0
fi

echo "[2/2] Running evaluation..."

deactivate

cd ../evaluation
source .venv/bin/activate

SUBMISSION_PATH="../outputs/${DATASET}/${OUTPUT_FILE}"
OUT_FILE_PATH="../results/${DATASET}/${OUTPUT_FILE}"

mkdir -p "../results/${DATASET}"

uv run python scoring_subtask_2.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[2/2] Evaluation complete."
echo "========================================"
echo "Results: ${OUT_FILE_PATH}"
