#!/bin/bash

#SBATCH --job-name=subtask1_inference
#SBATCH --output=../logs/inference_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

echo "========================================"
echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "========================================"

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

# Run script from its own directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Repository root
ROOT_DIR="$(git rev-parse --show-toplevel)"

# Activate repository-wide virtual environment
source "${ROOT_DIR}/.venv/bin/activate"

# Make shared common package importable
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Override at submission:
# sbatch --export=ALL,MODEL=google/medgemma-27b-it,PROMPT_INDEX=10 inference.sh

MODE="${MODE:-local}"                 # local / cloud
MODEL="${MODEL:-google/gemma-3-27b-it}"
DATASET="${DATASET:-dev}"             # dev / test
PROMPT_INDEX="${PROMPT_INDEX:-2}"

NUM_GPUS="${NUM_GPUS:-1}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Subtask1/data}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/Subtask1/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/Subtask1/results}"

RUN_EVAL="${RUN_EVAL:-true}"

# ============================================================================
# PATHS
# ============================================================================

XML_FILE="${DATA_ROOT}/${DATASET}/archehr-qa.xml"
KEY_FILE="${DATA_ROOT}/${DATASET}/archehr-qa_key.json"

OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"
RESULT_DIR="${RESULT_ROOT}/${DATASET}"

MODEL_NAME="$(echo "$MODEL" | tr '/' '-' | tr '.' '-')"

OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"

XML_PATH="$(realpath -m "$XML_FILE")"
KEY_PATH="$(realpath -m "$KEY_FILE")"

OUTPUT_PATH="$(realpath -m "${OUTPUT_DIR}/${OUTPUT_FILE}")"
RESULT_PATH="$(realpath -m "${RESULT_DIR}/${OUTPUT_FILE}")"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p ../logs

# ============================================================================
# VALIDATION
# ============================================================================

if [ ! -f "$XML_PATH" ]; then
    echo "ERROR: XML file not found:"
    echo "$XML_PATH"
    exit 1
fi

if [ ! -f "prompt.json" ]; then
    echo "ERROR: prompt.json not found"
    exit 1
fi

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# ============================================================================
# VLLM / TORCH SETTINGS
# ============================================================================

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_DISABLE=1

# ============================================================================
# RUN INFO
# ============================================================================

echo "Running inference"
echo "----------------------------------------"
echo "Mode          : $MODE"
echo "Model         : $MODEL"
echo "Dataset       : $DATASET"
echo "Prompt Index  : $PROMPT_INDEX"
echo "XML Path      : $XML_PATH"
echo "Output Path   : $OUTPUT_PATH"
echo "========================================"

# ============================================================================
# INFERENCE
# ============================================================================

PYTHONUNBUFFERED=1 python inference.py \
    --xml-file "$XML_PATH" \
    --prompt-file prompt.json \
    --prompt-index "$PROMPT_INDEX" \
    --output-file "$OUTPUT_PATH" \
    --inference-mode "$MODE" \
    --model "$MODEL"

echo "========================================"
echo "Inference complete."
echo "========================================"

# ============================================================================
# EVALUATION
# ============================================================================

if [ "$RUN_EVAL" = "true" ]; then

    if [ ! -f "$KEY_PATH" ]; then
        echo "Warning: Key file not found."
        echo "Skipping evaluation."
        exit 0
    fi

    echo "Running evaluation..."
    echo "========================================"

    cd ../evaluation

    SIF_IMAGE="./builder.sif"
    UV_BIN="$(which uv)"

    singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python evaluation.py \
        --submission_path "$OUTPUT_PATH" \
        --key_path "$KEY_PATH" \
        --quickumls_path ../quickumls/final \
        --out_file_path "$RESULT_PATH"

    echo "========================================"
    echo "Evaluation complete."
    echo "========================================"
    echo "Results saved to:"
    echo "$RESULT_PATH"

else

    echo "Evaluation skipped."

fi

echo "========================================"
echo "Job finished."
echo "========================================"