#!/bin/bash

#SBATCH --job-name=subtask2_synthetic
#SBATCH --output=../logs/synthetic%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia-rtx-a6000:1


# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# --- Configuration ---
XML_FILE="../../data/test_2025/archehr-qa.xml"
OUTPUT_FILE="../../data/test_2025/archehr-qa_synthetic_labels.json"
INFERENCE_MODE="local"
MODEL="qwen/qwen3-8b"

# Load .env file for API keys and HF token
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found"
fi

uv run python main.py \
    --xml-file "$XML_FILE" \
    --output-file "$OUTPUT_FILE" \
    --inference-mode "$INFERENCE_MODE" \
    --model "$MODEL"

echo "✓ Results saved to: $OUTPUT_FILE"
