#!/bin/bash
#SBATCH --job-name=subtask2_llm_dev
#SBATCH --output=../logs/batch%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Set the number of GPUs to use for both SLURM and tensor parallelism
NUM_GPUS=1
#SBATCH --gres=gpu:${NUM_GPUS}

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate


# Configurable variables
INFERENCE_MODE="local"  # local / cloud
MODEL="meta-llama/Llama-3.1-8B-Instruct"
    # meta-llama/Llama-3.1-8B-Instruct
    # google/medgemma-1.5-4b-it
DATASET="dev"  # dev / test / test-2026
PROMPT_INDEX=0

# Number of GPUs to use for tensor parallelism (should match SBATCH --gres)
TENSOR_PARALLEL_SIZE=$NUM_GPUS

# File Directories
DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"

# Auto-generate output filename: model_prompt_N.json
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"

# --- Environment Setup ---
uv run python inference.py \
    --xml-file ${DATA_DIR}/archehr-qa.xml \
    --prompt-file prompt.json \
    --prompt-index $PROMPT_INDEX \
    --output-file ${OUTPUT_DIR}/$OUTPUT_FILE \
    --inference-mode $INFERENCE_MODE \
    --model "$MODEL" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE


echo "[DONE] LLM labeling completed"
