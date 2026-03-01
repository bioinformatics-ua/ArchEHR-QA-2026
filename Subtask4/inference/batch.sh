#!/bin/bash
#SBATCH --job-name=subtask4_llm_dev
#SBATCH --output=../logs/subtask4_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4

# Num_GPUS must match --gres above and TENSOR_PARALLEL_SIZE in the script
NUM_GPUS=4

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# ----------------------------------------
# CONFIGURABLE VARIABLES
# ----------------------------------------

INFERENCE_MODE="cloud"   # local / cloud
PROMPT_INDEX=24
DATASET="dev"            # dev / test-2026
MODEL="google/gemini-2.5-flash"
    # Cloud Models:
    # google/gemini-2.5-flash
    # qwen/qwen3.5-flash-02-23
    # qwen/qwen3-max-thinking
    # anthropic/claude-opus-4.6
    # openai/gpt-5.2

    # Local Models:
    # Qwen/Qwen3.5-35B-A3B
    # google/medgemma-27b-text-it
    # google/medgemma-27b-it
    # google/medgemma-4b-it
    # khazarai/Bio-8B-it
    # Qwen/Qwen3-8B

# ----------------------------------------
# TWO-STEP INFERENCE OPTIONS
# ----------------------------------------

# Set to "twostep" to use inference_twostep.py, "standard" for inference.py,
# "prefixed" for inference_prefixed.py (N/A sentence ID prefixes)
INFERENCE_SCRIPT="standard"  # standard / twostep / prefixed

# Two-step flags (only apply when INFERENCE_SCRIPT="twostep"):
#   --no-second-pass               : skip second-pass verification (clinical knowledge filter only)
#   --no-clinical-knowledge-filter : skip clinical knowledge heuristic (second pass only)
#   (leave TWOSTEP_FLAGS empty to run full two-step pipeline)
TWOSTEP_FLAGS=""
# TWOSTEP_FLAGS="--no-second-pass"                    # filter only, no second pass
# TWOSTEP_FLAGS="--no-clinical-knowledge-filter"      # second pass only, no filter
# TWOSTEP_FLAGS="--no-second-pass --no-clinical-knowledge-filter"  # equivalent to standard


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
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"
OUTPUT_DIR="../outputs/${DATASET}"
RESULTS_DIR="../results/${DATASET}"

# Load model name for output file naming
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')

if [ "$INFERENCE_SCRIPT" = "twostep" ]; then
    OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_twostep.json"
    SCRIPT="inference_twostep.py"
elif [ "$INFERENCE_SCRIPT" = "prefixed" ]; then
    OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_prefixed.json"
    SCRIPT="inference_prefixed.py"
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
# RUN TEMPERATURES
# ----------------------------------------

for TEMP in 0.0 0.3 0.8
do
    echo "========================================"
    echo "Running inference (temperature=${TEMP})"
    echo "========================================"

    OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_t${TEMP}.json"

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
        --temperature $TEMP \
        --top-p $TOP_P \
        --max-tokens $MAX_TOKENS \
        --repetition-penalty $REPETITION_PENALTY \
        $TWOSTEP_FLAGS

done

echo "========================================"
echo "Running aggregation"
echo "========================================"

uv run python ../evaluation/aggregate_majority.py

# ----------------------------------------
# EVALUATE AGGREGATED
# ----------------------------------------

AGG_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_aggregated.json"

mv "../outputs/${DATASET}/aggregated_majority.json" \
   "../outputs/${DATASET}/${AGG_FILE}"

echo "========================================"
echo "Running evaluation on aggregated file"
echo "========================================"

deactivate
cd ../evaluation
source .venv/bin/activate

uv run python scoring_subtask_4.py \
    --submission_path "../outputs/${DATASET}/${AGG_FILE}" \
    --key_path "${KEY_PATH}" \
    --out_file_path "../results/${DATASET}/${AGG_FILE}"

echo "========================================"
echo "DONE"
echo "========================================"