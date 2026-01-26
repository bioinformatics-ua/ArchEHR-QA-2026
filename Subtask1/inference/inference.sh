#!/bin/bash

#SBATCH --job-name=subtask1_inference
#SBATCH --output=../logs/inference%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"


source .venv/bin/activate

# =============================================================================
# INFERENCE MODES:
# 1. LOCAL MODE: Uses vLLM with local GPU (default)
# 2. OPENAI MODE: Uses OpenAI API (requires OPENAI_API_KEY env variable)
# 3. GROQ MODE: Uses Groq API (requires GROQ_API_KEY env variable)
# =============================================================================

# --- Configuration ---
MODE="local"                         # Change to "local", "openai", or "groq"
MODEL="google/medgemma-27b-text-it"     # Full model name/path
DATASET="dev"                      # Change to "test" for test set or "dev" for development set
PROMPT_INDEX=8                      # Prompt template index

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

# --- Run the Inference Script ---
echo "Starting Python inference script..."
PYTHONUNBUFFERED=1 python inference.py \
    --xml-file ../../data/${DATASET}/archehr-qa.xml \
    --prompt-file prompt.json \
    --prompt-index $PROMPT_INDEX \
    --output-file ../outputs/${DATASET}/$OUTPUT_FILE \
    --inference-mode "$MODE" \
    --model "$MODEL"

# Deactivate inference venv before moving to evaluation
deactivate

cd ../evaluation 
SIF_IMAGE="./builder.sif"
UV_BIN=$(which uv)

# 1. Build dependencies (Fixes the C++ error)
# singularity exec --nv "$SIF_IMAGE" "$UV_BIN" sync

# 2. Run the code
singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python evaluation.py \
    --submission_path ../outputs/${DATASET}/$OUTPUT_FILE \
    --key_path ../../data/dev/archehr-qa.xml \
    --quickumls_path ../quickumls/final \
    --out_file_path ../results/${DATASET}/$OUTPUT_FILE

# echo "Evaluation complete. Job finished."
