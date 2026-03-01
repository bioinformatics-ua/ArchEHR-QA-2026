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
PROMPT_INDEX=5
DATASET="dev"            # dev / test-2026
MODEL="anthropic/claude-opus-4.6"
    # Cloud Models:
    # google/gemini-2.5-flash
    # google/gemini-3-flash-preview
    # google/gemini-2.5-pro
    # google/gemini-3-pro-preview
    # google/gemini-3.1-pro-preview
    # x-ai/grok-4.1-fast
    # x-ai/grok-4-fast
    # anthropic/claude-sonnet-4
    # anthropic/claude-sonnet-4.5
    # anthropic/claude-sonnet-4.6
    # anthropic/claude-opus-4.5
    # anthropic/claude-opus-4.6
    # moonshotai/kimi-k2-thinking
    # moonshotai/kimi-k2.5
    # qwen/qwen3.5-flash-02-23
    # qwen/qwen3-max-thinking

    # Local Models:
    # Qwen/Qwen3.5-35B-A3B
    # google/medgemma-27b-text-it
    # google/medgemma-27b-it
    # google/medgemma-4b-it
    # khazarai/Bio-8B-it
    # Qwen/Qwen3-8B

# ----------------------------------------
# INFERENCE SCRIPT OPTIONS
# ----------------------------------------

# Set to:
#   "standard"  : inference.py (default)
#   "twostep"   : inference_twostep.py
#   "prefixed"  : inference_prefixed.py (N/A sentence ID prefixes)
#   "ensemble"  : ensemble.py (no GPU needed, merges existing output files)
INFERENCE_SCRIPT="standard"

# Two-step flags (only apply when INFERENCE_SCRIPT="twostep"):
#   --no-second-pass               : skip second-pass verification (clinical knowledge filter only)
#   --no-clinical-knowledge-filter : skip clinical knowledge heuristic (second pass only)
TWOSTEP_FLAGS=""
# TWOSTEP_FLAGS="--no-second-pass"
# TWOSTEP_FLAGS="--no-clinical-knowledge-filter"
# TWOSTEP_FLAGS="--no-second-pass --no-clinical-knowledge-filter"

# ----------------------------------------
# ENSEMBLE OPTIONS (only used when INFERENCE_SCRIPT="ensemble")
# ----------------------------------------

# Strategy: union / intersection / majority
ENSEMBLE_STRATEGY="majority"

# Filenames of existing output files to ensemble (relative to OUTPUT_DIR)
ENSEMBLE_INPUTS=(
    "google-gemini-2-5-flash_prompt_4.json"
    "google-gemini-3-flash-preview_prompt_4.json"
    "anthropic-claude-sonnet-4-6_prompt_4.json"
    "x-ai-grok-4-fast_prompt_4.json"
)

# For majority strategy: minimum votes required (leave empty for ceil(n/2))
ENSEMBLE_MAJORITY_THRESHOLD="3"

# Output filename for the ensemble result (auto-generated or set manually)
# ENSEMBLE_OUTPUT_FILE="ensemble_union_p4_p5.json"
ENSEMBLE_OUTPUT_FILE="ensemble_${ENSEMBLE_STRATEGY}${ENSEMBLE_MAJORITY_THRESHOLD}_gemini25_gemini3_sonnet46_grok4_p4.json"


# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS  # Must match SBATCH --gres above
GPU_MEMORY_UTILIZATION=0.95     # VRAM fraction (0.3-0.4 shared, 0.85-0.95 dedicated)
MAX_MODEL_LEN=8192              # Context window in tokens

# --- Sampling ---
TEMPERATURE=0.0                 # Lower = more faithful/deterministic
TOP_P=0.95                      # Nucleus sampling cutoff (1.0 = disabled)
MAX_TOKENS=4096                 # Max tokens to generate per case
REPETITION_PENALTY=1.0          # >1.0 discourages repetitive phrasing


# ----------------------------------------
# PATHS
# ----------------------------------------
DATA_DIR="../../data/${DATASET}"
KEY_PATH="../../data-subtask2&3/${DATASET}/archehr-qa_key.json"
OUTPUT_DIR="../outputs/${DATASET}"
RESULTS_DIR="../results/${DATASET}"

# Model name for output file naming
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')

if [ "$INFERENCE_SCRIPT" = "twostep" ]; then
    OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_twostep.json"
    SCRIPT="inference_twostep.py"
elif [ "$INFERENCE_SCRIPT" = "prefixed" ]; then
    OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_prefixed.json"
    SCRIPT="inference_prefixed.py"
elif [ "$INFERENCE_SCRIPT" = "ensemble" ]; then
    OUTPUT_FILE="${ENSEMBLE_OUTPUT_FILE}"
    SCRIPT="ensemble.py"
else
    OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"
    SCRIPT="inference.py"
fi


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
# INFERENCE / ENSEMBLE
# ----------------------------------------

if [ "$INFERENCE_SCRIPT" = "ensemble" ]; then
    echo "[1/2] Running ensemble (strategy: ${ENSEMBLE_STRATEGY})..."

    # Build --inputs argument from array
    ENSEMBLE_INPUT_PATHS=()
    for f in "${ENSEMBLE_INPUTS[@]}"; do
        ENSEMBLE_INPUT_PATHS+=("${OUTPUT_DIR}/${f}")
    done

    MAJORITY_FLAG=""
    if [ -n "$ENSEMBLE_MAJORITY_THRESHOLD" ]; then
        MAJORITY_FLAG="--majority-threshold ${ENSEMBLE_MAJORITY_THRESHOLD}"
    fi

    uv run python ensemble.py \
        --inputs "${ENSEMBLE_INPUT_PATHS[@]}" \
        --strategy "$ENSEMBLE_STRATEGY" \
        --output "${OUTPUT_DIR}/${OUTPUT_FILE}" \
        $MAJORITY_FLAG

else
    echo "[1/2] Running inference with script: ${SCRIPT}..."

    #--debug-first-n 3 \

    uv run python $SCRIPT \
        --xml-file "${DATA_DIR}/archehr-qa.xml" \
        --qa-key-file "${KEY_PATH}" \
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
        --repetition-penalty $REPETITION_PENALTY \
        $TWOSTEP_FLAGS
fi

echo "[DONE] Subtask 4 inference/ensemble completed"


# ----------------------------------------
# SCORING (DEV ONLY)
# ----------------------------------------

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

uv run python scoring_subtask_4.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[2/2] Evaluation complete."