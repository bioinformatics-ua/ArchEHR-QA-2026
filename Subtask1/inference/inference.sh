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
DATASET="test"                      # Change to "test" for test set or "dev" for development set
PROMPT_INDEX=1                      # Prompt template index

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
python inference.py \
    --xml-file ../../data/${DATASET}/archehr-qa.xml \
    --prompt-file prompt.json \
    --prompt-index $PROMPT_INDEX \
    --output-file ../outputs/${DATASET}/$OUTPUT_FILE \
    --inference-mode $MODE \
    --model "$MODEL"
    # --model "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8" \ 


# {"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8" : "nemotron3"}
# nemotron3_prompt_1.json 



# # --- Run Evaluation Script ---
# echo "Starting evaluation script..."
# cd ../evaluation
# uv run python evaluation.py \
#     --submission_path ../outputs/predictions.json \
#     --key_path ../../data/dev/archehr-qa.xml \
#     --quickumls_path ../quickumls/ \
#     --out_file_path ../results/nemotron3_prompt_1_results.json


# echo "Evaluation complete. Job finished."
