#!/bin/bash

#SBATCH --job-name=subtask1_inference
#SBATCH --output=../logs/inference%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

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
MODE="cloud"                         # Change to "local", "openai", or "groq"
# MODEL="z-ai/glm-4.7"     # Full model name/path
MODEL="qwen/qwen3-235b-a22b-2507" # backup

DATASET="test"                      # Change to "test" for test set or "dev" for development set
PROMPT_INDEX=10                      # Prompt template index

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

# singularity exec --nv "$SIF_IMAGE" "$UV_BIN" sync


# echo "Evaluation complete. Job finished."
