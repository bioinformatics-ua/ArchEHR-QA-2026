#!/bin/bash
#SBATCH --job-name=subtask2_llm_dev
#SBATCH --output=../logs/batch%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2

# Set the number of GPUs to use for tensor parallelism
NUM_GPUS=2

# GPU Memory Configuration
# Adjust this value based on GPU availability (0.3-0.4 for shared GPUs, 0.85 for dedicated)
GPU_MEMORY_UTILIZATION=0.45

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate


# Configurable variables
INFERENCE_MODE="local"  # local / cloud
MODEL="meta-llama/Llama-3.1-8B-Instruct"
    # --- Models ---
    # meta-llama/Llama-3.1-8B-Instruct
        # Needs 2 gpu's with 0.45 GPU memory utilization
    # google/medgemma-1.5-4b-it
    # google/medgemma-4b-it
    # google/medgemma-27b-text-it
        # Needs 4 gpu's with 0.85-0.95 GPU memory utilization
    # google/medgemma-27b-it
    # google/gemma-3-27b-it
    # Qwen/Qwen3-32B
        # Needs 4 gpu's with 0.85-0.95 GPU memory utilization
DATASET="dev"  # dev / test / test-2026
PROMPT_INDEX=6

# Number of GPUs to use for tensor parallelism (should match SBATCH --gres)
TENSOR_PARALLEL_SIZE=$NUM_GPUS

# File Directories
DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"

# Auto-generate output filename: model_prompt_N.json
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"

# Load .env file for API keys and HF token
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found"
fi

# Disable PyTorch compilation to avoid Triton/Python.h issues
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_DISABLE=1

uv run python inference.py \
    --xml-file ${DATA_DIR}/archehr-qa.xml \
    --prompt-file prompt.json \
    --prompt-index $PROMPT_INDEX \
    --output-file ${OUTPUT_DIR}/$OUTPUT_FILE \
    --inference-mode $INFERENCE_MODE \
    --model "$MODEL" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION

echo "[DONE] LLM labeling completed"

cd ../evaluation 

# File Directories
SUBMISSION_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"
OUT_FILE_PATH="../results/${DATASET}/${OUTPUT_FILE}"

uv run python scoring_subtask_2.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[DONE] Subtask 2 scoring complete."