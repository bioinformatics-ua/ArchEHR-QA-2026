#!/bin/bash
#SBATCH --job-name=subtask2_union
#SBATCH --output=../logs/union_%j.out
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
MODEL="google/gemini-2.5-flash"
    # --- Cloud Models ---
    # google/gemini-2.5-flash
    # anthropic/claude-sonnet-4.5
    # anthropic/claude-sonnet-4.6
    # x-ai/grok-4.1-fast
PROMPT_INDICES_UNION="9 4"
PROMPT_INDEX_CRITIC="3"
    # deepseek/deepseek-v3.2
    # qwen/qwen3-max-thinking
    # google/gemini-3-flash-preview



# --- GPU / Engine ---
TENSOR_PARALLEL_SIZE=$NUM_GPUS
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=4096

# --- Sampling ---
TEMPERATURE=0.0
TOP_P=0.95
MAX_TOKENS=512
REPETITION_PENALTY=1.0

# --- File Directories ---
DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"

# Auto-generate output filename  e.g. google-gemini-2-5-flash_union_p9-p4.json
MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
INDICES_TAG=$(echo "$PROMPT_INDICES" | tr ' ' '-' | sed 's/-/p/g' | sed 's/^/p/')
OUTPUT_FILE="${MODEL_NAME}_union_${INDICES_TAG}.json"

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
# UNION INFERENCE (Step 1: union 9+4, Step 2: filter with 3)
# =============================================================================
echo "[1/4] Running union inference (prompts: ${PROMPT_INDICES_UNION})..."

MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
INDICES_TAG_UNION=$(echo "$PROMPT_INDICES_UNION" | tr ' ' '-' | sed 's/-/p/g' | sed 's/^/p/')
OUTPUT_FILE_UNION="${MODEL_NAME}_union_${INDICES_TAG_UNION}.json"

uv run python union.py \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --prompt-file prompt.json \
    --prompt-indices $PROMPT_INDICES_UNION \
    --output-file "${OUTPUT_DIR}/${OUTPUT_FILE_UNION}" \
    --inference-mode "$INFERENCE_MODE" \
    --model "$MODEL" \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-tokens $MAX_TOKENS \
    --repetition-penalty $REPETITION_PENALTY \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN

echo "[2/4] Union inference complete."

echo "[3/4] Running critic/filter (prompt: ${PROMPT_INDEX_CRITIC})..."

INDICES_TAG_FINAL="p9-p4-p3"
OUTPUT_FILE_FINAL="${MODEL_NAME}_union_${INDICES_TAG_FINAL}.json"

uv run python union.py \
    --xml-file "${DATA_DIR}/archehr-qa.xml" \
    --prompt-file prompt.json \
    --prompt-indices $PROMPT_INDEX_CRITIC \
    --filter-predictions "${OUTPUT_DIR}/${OUTPUT_FILE_UNION}" \
    --output-file "${OUTPUT_DIR}/${OUTPUT_FILE_FINAL}" \
    --inference-mode "$INFERENCE_MODE" \
    --model "$MODEL" \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-tokens $MAX_TOKENS \
    --repetition-penalty $REPETITION_PENALTY \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN

echo "[4/4] Critic/filter step complete."

# =============================================================================
# EVALUATION
# =============================================================================

KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"

if [ ! -f "${KEY_PATH}" ]; then
    echo "[5/5] No key file found for ${DATASET}, skipping evaluation."
    echo "========================================"
    echo "Output: ${OUTPUT_DIR}/${OUTPUT_FILE_FINAL}"
    exit 0
fi

echo "[5/5] Running evaluation..."

deactivate

cd ../evaluation
source .venv/bin/activate

SUBMISSION_PATH="../outputs/${DATASET}/${OUTPUT_FILE_FINAL}"
OUT_FILE_PATH="../results/${DATASET}/${OUTPUT_FILE_FINAL}"

mkdir -p "../results/${DATASET}"

uv run python scoring_subtask_2.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[5/5] Evaluation complete."

# =============================================================================
# RESULTS ANALYSIS (per-sentence P/R/F1)
# =============================================================================
echo "[6/6] Running per-sentence analysis..."

ANALYSIS_OUT_PATH="../results-analysis/${DATASET}/${OUTPUT_FILE_FINAL}"
mkdir -p "../results-analysis/${DATASET}"

uv run python results_analysis.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --xml_path "../../data/${DATASET}/archehr-qa.xml" \
    --out_file_path "$ANALYSIS_OUT_PATH"

echo "[6/6] Analysis complete."
echo "========================================"
echo "Results      : ${OUT_FILE_PATH}"
echo "Per-sentence : ${ANALYSIS_OUT_PATH}"
