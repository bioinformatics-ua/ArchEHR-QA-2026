#!/bin/bash
#SBATCH --job-name=subtask3_inference
#SBATCH --output=../logs/inference_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
    
NUM_GPUS=1  # Must match SBATCH --gres gpu count

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

# --- Task ---
INFERENCE_MODE="local"      # local / cloud
DATASET="test"               # dev / test / test-2026
PROMPT_INDEX=1
MODEL="meta-llama/Llama-3.1-8B-Instruct"
    # --- Local Models ---
    # meta-llama/Llama-3.1-8B-Instruct
    # google/medgemma-27b-text-it
    # google/gemma-3-27b-it
    # Qwen/Qwen3-32B
    # Qwen/Qwen3-8B
    # zai-org/GLM-4.6V-Flash
    # Echelon-AI/Med-Qwen2-7B

    # --- Cloud Models ---
    # anthropic/claude-sonnet-4.5
    # anthropic/claude-opus-4.6
    # deepseek/deepseek-v3.2
    # x-ai/grok-4.1-fast
    # moonshotai/kimi-k2.5
    # minimax/minimax-m2.5
    # openai/gpt-5-mini
    # z-ai/glm-4.6v
    # qwen/qwen3-max-thinking
    # google/gemini-3-flash-preview
    # google/gemini-2.5-flash
    # openai/gpt-4.1
    # openai/gpt-5.2
    # z-ai/glm-5

# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS  # Must match SBATCH --gres above
GPU_MEMORY_UTILIZATION=0.95     # VRAM fraction (0.3-0.4 shared, 0.85-0.95 dedicated)
MAX_MODEL_LEN=4096              # 150 for non-thinking models. Context window in tokens (increase for long notes)

# --- Sampling ---
TEMPERATURE=0.0                 # Lower = more faithful/deterministic, higher = more diverse
TOP_P=0.95                      # Nucleus sampling cutoff (1.0 = disabled)
MAX_TOKENS=4096                 # 150 for non-thinking models. Max tokens to generate per answer
REPETITION_PENALTY=1.0         # >1.0 discourages repetitive phrasing

# --- File Directories ---
DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"
OUTPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"
ANALYSIS_DIR="${SUBTASK_DIR}/analysis/${DATASET}"
LOGS_DIR="${SUBTASK_DIR}/logs"
PROMPT_FILE="${SUBTASK_DIR}/inference/prompt.json"

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}" "${ANALYSIS_DIR}" "${LOGS_DIR}"

# Auto-generate output filename
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"

# Load .env for HF_TOKEN / OPENROUTER_API_KEY
if [ -f "${SUBTASK_DIR}/inference/.env" ]; then
    export $(cat "${SUBTASK_DIR}/inference/.env" | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# Disable PyTorch compilation to avoid Triton/Python.h issues
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_DISABLE=1

# =============================================================================
# INFERENCE
# =============================================================================
echo "[1/2] Running inference..."

uv run python "${SUBTASK_DIR}/inference/inference.py" \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --prompt-file "${PROMPT_FILE}" \
    --prompt-index $PROMPT_INDEX \
    --output-file "${OUTPUT_DIR}/${OUTPUT_FILE}" \
    --inference-mode "$INFERENCE_MODE" \
    --model "$MODEL" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-tokens $MAX_TOKENS \
    --repetition-penalty $REPETITION_PENALTY

echo "[1/2] Inference complete."

# =============================================================================
# EVALUATION
# =============================================================================
echo "[2/2] Running evaluation..."

deactivate

cd ../evaluation

SIF_IMAGE="./builder.sif"
UV_BIN=$(which uv)

SUBMISSION_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"
KEY_PATH="${DATA_DIR}/archehr-qa_key.json"
DATA_PATH="${DATA_DIR}/archehr-qa.xml"
OUT_FILE_PATH="${RESULTS_DIR}/${OUTPUT_FILE}"

singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python scoring_subtask_3.py \
     --submission_path "$SUBMISSION_PATH" \
     --key_path "$KEY_PATH" \
     --data_path "$DATA_PATH" \
     --quickumls_path ./quickumls/final \
     --out_file_path "$OUT_FILE_PATH"

echo "[2/2] Evaluation complete."
echo "========================================"
echo "Results: ${OUT_FILE_PATH}"
