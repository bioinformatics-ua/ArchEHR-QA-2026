#!/bin/bash
#SBATCH --job-name=subtask4_llm_dev
#SBATCH --output=../logs/subtask4_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# exclude=ailab-l4-01,ailab-l4-02,ailab-l4-05,ailab-l4-06,ailab-l4-08,ailab-l4-09,ailab-l4-10,ailab-l4-11

# Num_GPUS must match --gres above and TENSOR_PARALLEL_SIZE in the script
NUM_GPUS=1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# ----------------------------------------
# CONFIGURABLE VARIABLES
# ----------------------------------------

INFERENCE_MODE="local"   # local / cloud
PROMPT_INDEX=10
DATASET="dev"            # dev / test-2026
MODEL="meta-llama/Llama-3.1-8B-Instruct"

    # --- Local Models ---
    # google/medgemma-27b-text-it           # DONE
    # google/medgemma-4b-it                 # DONE
    # google/gemma-3-27b-it                 # DONE
    # google/medgemma-1.5-4b-it             # DONE
    # khazarai/Bio-8B-it                    # DONE
    # Qwen/Qwen3-8B                         # DONE
    # Qwen/Qwen3-32b                        # DONE
    # BioMistral/BioMistral-7B              # DONE
    # Echelon-AI/Med-Qwen2-7B               # DONE
    # meta-llama/Llama-3.1-8B-Instruct      # DONE

    # --- Cloud Models ---
    # google/gemini-2.5-flash
    # google/gemini-3-flash-preview
    # google/gemini-2.5-flash-lite
    # google/gemini-2.5-flash-lite-preview-09-2025
    # google/gemini-2.0-flash-001
    # google/gemini-2.5-pro
    # google/gemini-3-pro-preview
    # google/gemini-3.1-pro-preview
    # x-ai/grok-4.1-fast
    # x-ai/grok-4-fast
    # x-ai/grok-4
    # openai/gpt-5.2
    # openai/gpt-5.1
    # openai/gpt-5
    # openai/gpt-5-mini
    # openai/gpt-5-nano
    # openai/gpt-oss-20b                                            # not tried
    # openai/gpt-4.1
    # openai/gpt-4.1-mini
    # deepseek/deepseek-v3.2
    # deepseek/deepseek-chat-v3-0324
    # anthropic/claude-sonnet-4
    # anthropic/claude-sonnet-4.5
    # anthropic/claude-sonnet-4.6
    # anthropic/claude-opus-4.5
    # anthropic/claude-opus-4.6
    # anthropic/claude-haiku-4.5
    # moonshotai/kimi-k2-thinking                                   # not tried (prompt_5)
    # moonshotai/kimi-k2.5                                          # not tried (prompt_5)
    # moonshotai/kimi-k2-0905                                       # not tried
    # qwen/qwen3.5-flash-02-23
    # qwen/qwen3-max-thinking
    # qwen/qwen3.5-plus-02-15
    # qwen/qwen3.5-397b-a17b
    # qwen/qwen3-30b-a3b-thinking-2507                              # not tried
    # qwen/qwen3-235b-a22b-2507
    # qwen/qwen3-32b
    # z-ai/glm-5                                                    # not tried
    # z-ai/glm-4.7-flash
    # z-ai/glm-4.7                                                  # not tried
    # minimax/minimax-m2.5
    # minimax/minimax-m2.1
    # xiaomi/mimo-v2-flash
    # nvidia/nemotron-3-nano-30b-a3b:free
    # meta-llama/llama-4-scout
    # meta-llama/llama-3.3-70b-instruct

    # Submission 1:
    # google/gemini-2.5-flash
    # anthropic/claude-opus-4.6
    # google/gemini-2.0-flash-001

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
# - union: include all unique answers from all models
# - intersection: include only answers that all models agree on
# - majority: include answers that meet the majority threshold (e.g. 2 out of 3)
ENSEMBLE_STRATEGY="majority"

# Fallback model file (leave empty for no fallback)
ENSEMBLE_FALLBACK="openai-gpt-4-1_prompt_5.json"

# Filenames of existing output files to ensemble (relative to OUTPUT_DIR)
ENSEMBLE_INPUTS=(
    "google-gemini-2-5-flash_prompt_5.json"
    "google-gemini-2-5-flash_prompt_6.json"
    "openai-gpt-4-1_prompt_6.json"
    # "x-ai-grok-4-fast_prompt_5.json"
    # "openai-gpt-5_prompt_7.json"
    # "google-gemini-2-0-flash-001_prompt_5.json"
    # "google-gemini-2-5-flash_prompt_6.json"
    # "anthropic-claude-opus-4-5_prompt_5.json"
    # "anthropic-claude-opus-4-6_prompt_5.json"
    # "anthropic-claude-sonnet-4-5_prompt_5.json"
    # "google-gemini-3-flash-preview_prompt_5.json"
    # "anthropic-claude-sonnet-4_prompt_5.json"
    # "x-ai-grok-4-1-fast_prompt_5.json"
    # "nvidia-nemotron-3-nano-30b-a3b:free_prompt_5.json"
    # "deepseek-deepseek-v3-2_prompt_5.json"
    # "qwen-qwen3-max-thinking_prompt_5.json"
)

# For majority strategy: minimum votes required (leave empty for ceil(n/2))
ENSEMBLE_MAJORITY_THRESHOLD="2"

# Output filename for the ensemble result (auto-generated or set manually)
# ENSEMBLE_OUTPUT_FILE="ensemble_union_p4_p5.json"
ENSEMBLE_OUTPUT_FILE="ensemble_${ENSEMBLE_STRATEGY}${ENSEMBLE_MAJORITY_THRESHOLD}_4_models.json"


# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS  # Must match SBATCH --gres above
GPU_MEMORY_UTILIZATION=0.95     # VRAM fraction (0.3-0.4 shared, 0.85-0.95 dedicated)
MAX_MODEL_LEN=8192              # Context window in tokens

# --- Sampling ---
TEMPERATURE=0.0                 # Lower = more faithful/deterministic
TOP_P=0.95                      # Nucleus sampling cutoff (1.0 = disabled)
MAX_TOKENS=8192                 # Max tokens to generate per case
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
        ${ENSEMBLE_MAJORITY_THRESHOLD:+--majority-threshold "$ENSEMBLE_MAJORITY_THRESHOLD"} \
        ${ENSEMBLE_FALLBACK:+--fallback "${OUTPUT_DIR}/${ENSEMBLE_FALLBACK}"}

else
    echo "[1/2] Running inference with script: ${SCRIPT}..."

    #--debug-first-n 3 \

    # uv run --no-sync python $SCRIPT \
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