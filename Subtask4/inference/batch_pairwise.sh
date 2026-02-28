#!/bin/bash
#SBATCH --job-name=subtask4_llm_dev
#SBATCH --output=../logs/subtask4_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Num_GPUS must match --gres above and TENSOR_PARALLEL_SIZE in the script
NUM_GPUS=1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# ----------------------------------------
# CONFIGURABLE VARIABLES
# ----------------------------------------

INFERENCE_MODE="cloud"   # local / cloud
PROMPT_INDEX=3           # <--- CHANGED TO 3 FOR THE PAIRWISE PROMPT
DATASET="dev"            # dev / test-2026
MODEL="google/gemini-2.5-flash"


# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS  # Must match SBATCH --gres above
GPU_MEMORY_UTILIZATION=0.25     # VRAM fraction (0.3-0.4 shared, 0.85-0.95 dedicated)
MAX_MODEL_LEN=4096              # Context window in tokens

# --- Sampling ---
TEMPERATURE=0.0                 # Lower = more faithful/deterministic
TOP_P=0.95                      # Nucleus sampling cutoff (1.0 = disabled)
MAX_TOKENS=512                  # Max tokens to generate per case
REPETITION_PENALTY=1.0          # >1.0 discourages repetitive phrasing


# ----------------------------------------
# PATHS
# ----------------------------------------
DATA_DIR="../../data/${DATASET}"
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"
OUTPUT_DIR="../outputs/${DATASET}"
RESULTS_DIR="../results/${DATASET}"

# Load model name for output file naming
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"


# ----------------------------------------
# LOAD ENV
# ----------------------------------------
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "ERROR: .env file not found (cloud requires API keys)"
    exit 1
fi

# ----------------------------------------
# INFERENCE
# ----------------------------------------
echo "[1/2] Running inference..."

uv run python inference_subtask4_pairwise.py \
    --debug-first-n 3 \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --qa-key-file "${KEY_PATH}" \
    --prompt-file prompt_subtask4.json \
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

echo "[DONE] Subtask 4 inference completed"


# ----------------------------------------
# SCORING (DEV ONLY)
# ----------------------------------------

if [ ! -f "${KEY_PATH}" ]; then
    echo "[2/3] No key file found for ${DATASET}, skipping evaluation and analysis."
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

uv run python scoring_subtask_4.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[2/2] Evaluation complete."