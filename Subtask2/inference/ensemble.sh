#!/bin/bash
#SBATCH --job-name=subtask2_ensemble
#SBATCH --output=/ceph/home/student.aau.dk/lj02sb/ArchEHR-QA-2026/Subtask2/logs/ensemble_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

NUM_GPUS=1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# =============================================================================
# ROOT + ENVIRONMENT
# =============================================================================

ROOT_DIR=$(git rev-parse --show-toplevel)

cd "$ROOT_DIR"

source "${ROOT_DIR}/.venv/bin/activate"

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Task ---
INFERENCE_MODE="cloud"
DATASET="dev"
PROMPT_INDEX=7
MODEL="anthropic/claude-sonnet-4.5"

# =============================================================================
# ENSEMBLE SETTINGS
# =============================================================================

N_RUNS=15
TEMPERATURE=0.3
MIN_VOTES=2

# =============================================================================
# ENGINE SETTINGS
# =============================================================================

TENSOR_PARALLEL_SIZE=$NUM_GPUS
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=4096

# =============================================================================
# SAMPLING
# =============================================================================

TOP_P=0.95
MAX_TOKENS=512
REPETITION_PENALTY=1.0

# =============================================================================
# PATHS
# =============================================================================

SUBTASK_DIR="${ROOT_DIR}/Subtask2"

DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"

OUTPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"

RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"

ANALYSIS_DIR="${SUBTASK_DIR}/results-analysis/${DATASET}"

PROMPT_FILE="${SUBTASK_DIR}/inference/prompt.json"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${ANALYSIS_DIR}"
mkdir -p "${SUBTASK_DIR}/logs"

# =============================================================================
# OUTPUT FILE
# =============================================================================

MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')

TEMP_TAG=$(echo "$TEMPERATURE" | tr '.' '-')

OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_ensemble${N_RUNS}x_t${TEMP_TAG}_votes${MIN_VOTES}.json"

# =============================================================================
# LOAD ENV VARIABLES
# =============================================================================

if [ -f "${ROOT_DIR}/.env" ]; then
    export $(cat "${ROOT_DIR}/.env" | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# =============================================================================
# VLLM / TORCH SETTINGS
# =============================================================================

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_DISABLE=1

# =============================================================================
# ENSEMBLE INFERENCE
# =============================================================================

echo "[1/3] Running ensemble inference..."

uv run python -m Subtask2.inference.ensemble \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --prompt-file "${PROMPT_FILE}" \
    --prompt-index $PROMPT_INDEX \
    --output-file "${OUTPUT_DIR}/${OUTPUT_FILE}" \
    --inference-mode "$INFERENCE_MODE" \
    --model "$MODEL" \
    --n-runs $N_RUNS \
    --min-votes $MIN_VOTES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-tokens $MAX_TOKENS \
    --repetition-penalty $REPETITION_PENALTY \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN

echo "[1/3] Ensemble inference complete."

# =============================================================================
# EVALUATION
# =============================================================================

KEY_PATH="${DATA_DIR}/archehr-qa_key.json"

if [ ! -f "${KEY_PATH}" ]; then
    echo "[2/3] No key file found for ${DATASET}"
    echo "Skipping evaluation and analysis."
    echo "========================================"
    echo "Output: ${OUTPUT_DIR}/${OUTPUT_FILE}"
    exit 0
fi

echo "[2/3] Running evaluation..."

SUBMISSION_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"

OUT_FILE_PATH="${RESULTS_DIR}/${OUTPUT_FILE}"

uv run python "${SUBTASK_DIR}/evaluation/scoring_subtask_2.py" \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[2/3] Evaluation complete."

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

echo "[3/3] Running per-sentence analysis..."

ANALYSIS_OUT_PATH="${ANALYSIS_DIR}/${OUTPUT_FILE}"

uv run python "${SUBTASK_DIR}/evaluation/results_analysis.py" \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --xml_path "${DATA_DIR}/archehr-qa.xml" \
    --out_file_path "$ANALYSIS_OUT_PATH"

echo "[3/3] Analysis complete."

echo "========================================"
echo "Results      : ${OUT_FILE_PATH}"
echo "Analysis     : ${ANALYSIS_OUT_PATH}"
echo "Submission   : ${SUBMISSION_PATH}"