#!/bin/bash
#SBATCH --job-name=subtask2_inference
#SBATCH --output=../logs/inference_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

NUM_GPUS=1  # Must match SBATCH --gres gpu count
# You need minimum 1 gpu to run evaluation after inference if you use a cloud model

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Task ---
INFERENCE_MODE="cloud"      # local / cloud
DATASET="dev"               # dev / test / test-2026
PROMPT_INDEX=7
MODEL="anthropic/claude-sonnet-4.5"
    # --- Local Models ---
    # meta-llama/Llama-3.1-8B-Instruct
    # google/medgemma-27b-text-it
        # 4 gpu's, 0.95 GPU memory utilization
    # google/gemma-3-27b-it
        # 4 gpu's, 0.95 GPU memory utilization
    # Qwen/Qwen3-32B
        # 4 gpu's, 0.95 GPU memory utilization
    # Qwen/Qwen3-8B
        # 1 gpu, 0.95 GPU memory utilization

    # --- Cloud Models ---
    # anthropic/claude-sonnet-4.5
    # anthropic/claude-opus-4.6
    # deepseek/deepseek-v3.2
    # x-ai/grok-4.1-fast
    # google/gemini-2.5-flash
    # openai/gpt-4.1
    # openai/gpt-5-mini
    # qwen/qwen3-max-thinking
    # anthropic/claude-sonnet-4.6

# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS  # Must match SBATCH --gres above
GPU_MEMORY_UTILIZATION=0.95     # VRAM fraction (0.3-0.4 shared, 0.85-0.95 dedicated)
MAX_MODEL_LEN=4096              # Context window in tokens

# --- Sampling ---
TEMPERATURE=0.5                 # Lower = more faithful/deterministic
TOP_P=0.95                      # Nucleus sampling cutoff (1.0 = disabled)
MAX_TOKENS=512                  # Max tokens to generate per case
REPETITION_PENALTY=1.0          # >1.0 discourages repetitive phrasing

# --- File Directories ---
DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"

# Auto-generate output filename
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"

# Load .env for HF_TOKEN / OPENROUTER_API_KEY
if [ -f .env ]; then
    export $(cat .env | xargs)
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

uv run python inference.py \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --prompt-file prompt.json \
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
