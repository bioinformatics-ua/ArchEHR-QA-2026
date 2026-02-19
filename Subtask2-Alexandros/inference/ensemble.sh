#!/bin/bash
#SBATCH --job-name=subtask2_ensemble
#SBATCH --output=../logs/ensemble_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

NUM_GPUS=1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Task ---
INFERENCE_MODE="cloud"
DATASET="dev"               # dev / test / test-2026
PROMPT_INDEX=7
MODEL="anthropic/claude-sonnet-4.5"
    # --- Cloud Models ---
    # anthropic/claude-sonnet-4.5
    # anthropic/claude-sonnet-4.6
    # x-ai/grok-4.1-fast
    # openai/gpt-4.1
    # deepseek/deepseek-v3.2
    # qwen/qwen3-max-thinking
    # google/gemini-3-flash-preview
    # google/gemini-2.5-flash
    # anthropic/claude-sonnet-4
    # x-ai/grok-4-fast

    # --- Local Models ---
    # meta-llama/Llama-3.1-8B-Instruct
    # google/medgemma-27b-text-it
    # google/gemma-3-27b-it
    # google/medgemma-1.5-4b-it

# --- Ensemble settings ---
N_RUNS=15                    # Number of independent inference passes
TEMPERATURE=0.3             # Higher temperature → more diversity between runs
MIN_VOTES=2                 # Minimum runs a sentence must appear in to be included (1 = union, N_RUNS = intersection)

# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=4096

# --- Sampling ---
TOP_P=0.95
MAX_TOKENS=512
REPETITION_PENALTY=1.0

# --- File Directories ---
DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"

# Auto-generate output filename  e.g. google-gemini-2-5-flash_prompt_5_ensemble5x_t0-7.json
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
TEMP_TAG=$(echo "$TEMPERATURE" | tr '.' '-')
OUTPUT_FILE="${MODEL_NAME}_prompt_${PROMPT_INDEX}_ensemble${N_RUNS}x_t${TEMP_TAG}_votes${MIN_VOTES}.json"

# Load .env
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_TRITON_FLASH_ATTN=0
export TORCH_COMPILE_DISABLE=1

# =============================================================================
# ENSEMBLE INFERENCE
# =============================================================================
echo "[1/3] Running ensemble inference (${N_RUNS} runs × prompt ${PROMPT_INDEX})..."

uv run python ensemble.py \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --prompt-file prompt.json \
    --prompt-index $PROMPT_INDEX \
    --output-file "${OUTPUT_DIR}/${OUTPUT_FILE}" \
    --inference-mode "$INFERENCE_MODE" \
    --model "$MODEL" \
    --n-runs $N_RUNS \
    --min-votes $MIN_VOTES \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-tokens $MAX_TOKENS \
    --repetition-penalty $REPETITION_PENALTY \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN

echo "[1/3] Ensemble inference complete."

# =============================================================================
# EVALUATION
# =============================================================================
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"

if [ ! -f "${KEY_PATH}" ]; then
    echo "[2/3] No key file found for ${DATASET}, skipping evaluation."
    echo "========================================"
    echo "Output: ${OUTPUT_DIR}/${OUTPUT_FILE}"
    exit 0
fi

echo "[2/3] Running evaluation..."

deactivate

cd ../evaluation
source .venv/bin/activate

SUBMISSION_PATH="../outputs/${DATASET}/${OUTPUT_FILE}"
OUT_FILE_PATH="../results/${DATASET}/${OUTPUT_FILE}"

mkdir -p "../results/${DATASET}"

uv run python scoring_subtask_2.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[2/3] Evaluation complete."

# =============================================================================
# RESULTS ANALYSIS (per-sentence P/R/F1)
# =============================================================================
echo "[3/3] Running per-sentence analysis..."

ANALYSIS_OUT_PATH="../results-analysis/${DATASET}/${OUTPUT_FILE}"
mkdir -p "../results-analysis/${DATASET}"

uv run python results_analysis.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --xml_path "../../data/${DATASET}/archehr-qa.xml" \
    --out_file_path "$ANALYSIS_OUT_PATH"

echo "[3/3] Analysis complete."
echo "========================================"
echo "Results      : ${OUT_FILE_PATH}"
echo "Per-sentence : ${ANALYSIS_OUT_PATH}"
