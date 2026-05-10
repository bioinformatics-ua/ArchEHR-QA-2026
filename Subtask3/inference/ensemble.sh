#!/bin/bash
#SBATCH --job-name=subtask3_ensemble
#SBATCH --output=../logs/ensemble_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
SUBTASK_DIR="${ROOT_DIR}/Subtask3"

source "${ROOT_DIR}/.venv/bin/activate"

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET="test-2026"               # dev / test / test-2026
JUDGE_MODEL="openai/gpt-4.1"
MODE="select"                # select = pick best candidate; merge = synthesise
TEMPERATURE=0.0
MAX_TOKENS=4096
FALLBACK=0                   # 0-based index of fallback candidate file

# --- Candidate files to ensemble (best runs, ranked by overall_score) ---
# Adjust these to your top-performing runs.
DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"
OUTPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"
ANALYSIS_DIR="${SUBTASK_DIR}/analysis/${DATASET}"
LOGS_DIR="${SUBTASK_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}" "${ANALYSIS_DIR}" "${LOGS_DIR}"

CANDIDATES=(
    "${OUTPUT_DIR}/anthropic-claude-sonnet-4-5_prompt_6.json"
    "${OUTPUT_DIR}/anthropic-claude-opus-4-6_prompt_6.json"
    "${OUTPUT_DIR}/google-gemini-2-5-flash_prompt_6.json"
    #"${OUTPUT_DIR}/anthropic-claude-sonnet-4-5_prompt_11.json"
    #"${OUTPUT_DIR}/google-gemini-3-flash-preview_prompt_6.json"
    #"${OUTPUT_DIR}/openai-gpt-4-1_prompt_6.json"
)

# --- Output filename ---
OUTPUT_FILE="ensemble_${MODE}_$(echo $JUDGE_MODEL | tr '/' '-' | tr '.' '-')_top${#CANDIDATES[@]}.json"

# Load .env for API keys
if [ -f "${SUBTASK_DIR}/inference/.env" ]; then
    export $(cat "${SUBTASK_DIR}/inference/.env" | xargs)
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

uv run python "${SUBTASK_DIR}/inference/ensemble.py" \
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
echo "[2/2] Running evaluation..."

deactivate

cd ../evaluation

#SIF_IMAGE="./builder.sif"
#UV_BIN=$(which uv)

SUBMISSION_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"
KEY_PATH="${DATA_DIR}/archehr-qa_key.json"
DATA_PATH="${DATA_DIR}/archehr-qa.xml"
OUT_FILE_PATH="${RESULTS_DIR}/${OUTPUT_FILE}"

#singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python scoring_subtask_3.py \
#     --submission_path "$SUBMISSION_PATH" \
#     --key_path "$KEY_PATH" \
#     --data_path "$DATA_PATH" \
#     --quickumls_path ./quickumls/final \
#     --out_file_path "$OUT_FILE_PATH"

echo "[2/2] Evaluation complete."
echo "========================================"
echo "Results: ${OUT_FILE_PATH}"
