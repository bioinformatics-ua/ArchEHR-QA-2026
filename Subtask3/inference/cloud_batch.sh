#!/bin/bash
#SBATCH --job-name=subtask3_cloud_batch
#SBATCH --output=../logs/cloud_batch_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

NUM_GPUS=1
DATASET="test"
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=4096
TEMPERATURE=0.0
TOP_P=0.95
MAX_TOKENS=4096
REPETITION_PENALTY=1.0

RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
SUBTASK_DIR="${ROOT_DIR}/Subtask3"

DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"
OUTPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"
ANALYSIS_DIR="${SUBTASK_DIR}/analysis/${DATASET}"
LOGS_DIR="${SUBTASK_DIR}/logs"
PROMPT_FILE="${SUBTASK_DIR}/inference/prompt.json"

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}" "${ANALYSIS_DIR}" "${LOGS_DIR}"

#    "anthropic/claude-sonnet-4.5"

#    "deepseek/deepseek-v3.2"
#    "x-ai/grok-4.1-fast"
#    "openai/gpt-5-mini"

# List of closed source models (exact names)
CLOUD_MODELS=(
    "google/gemini-2.5-flash"
    "openai/gpt-4.1"
)

source "${ROOT_DIR}/.venv/bin/activate"

# Load .env for HF_TOKEN / OPENROUTER_API_KEY
if [ -f "${SUBTASK_DIR}/inference/.env" ]; then
    export $(cat "${SUBTASK_DIR}/inference/.env" | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

for MODEL in "${CLOUD_MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
    for PROMPT_INDEX in {1..11}; do
        echo "Processing $MODEL (prompt $PROMPT_INDEX)"
        uv run python "${SUBTASK_DIR}/inference/inference.py" \
            --xml-file "${DATA_DIR}/archehr-qa.xml" \
            --prompt-file "$PROMPT_FILE" \
            --prompt-index $PROMPT_INDEX \
            --output-file "${OUTPUT_DIR}/${MODEL_NAME}_prompt_${PROMPT_INDEX}.json" \
            --inference-mode "cloud" \
            --model "$MODEL" \
            --tensor-parallel-size $NUM_GPUS \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --max-model-len $MAX_MODEL_LEN \
            --temperature $TEMPERATURE \
            --top-p $TOP_P \
            --max-tokens $MAX_TOKENS \
            --repetition-penalty $REPETITION_PENALTY

        echo "Running evaluation for $MODEL (prompt $PROMPT_INDEX)"
        deactivate
        cd ../evaluation
        SIF_IMAGE="./builder.sif"
        UV_BIN=$(which uv)
        SUBMISSION_PATH="${OUTPUT_DIR}/${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"
        KEY_PATH="${DATA_DIR}/archehr-qa_key.json"
        DATA_PATH="${DATA_DIR}/archehr-qa.xml"
        OUT_FILE_PATH="${RESULTS_DIR}/${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"
        singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python scoring_subtask_3.py \
            --submission_path "$SUBMISSION_PATH" \
            --key_path "$KEY_PATH" \
            --data_path "$DATA_PATH" \
            --quickumls_path ./quickumls/final \
            --out_file_path "$OUT_FILE_PATH"
        deactivate
        cd ../inference
    done
done
