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

DATA_DIR="../../data/${DATASET}"
OUTPUT_DIR="../outputs/${DATASET}"
RESULTS_DIR="../results/${DATASET}"
PROMPT_FILE="prompt.json"

#    "anthropic/claude-sonnet-4.5"

#    "deepseek/deepseek-v3.2"
#    "x-ai/grok-4.1-fast"
#    "openai/gpt-5-mini"

# List of closed source models (exact names)
CLOUD_MODELS=(
    "google/gemini-2.5-flash"
    "openai/gpt-4.1"
)

source .venv/bin/activate

# Load .env for HF_TOKEN / OPENROUTER_API_KEY
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

for MODEL in "${CLOUD_MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
    for PROMPT_INDEX in {1..11}; do
        echo "Processing $MODEL (prompt $PROMPT_INDEX)"
        uv run python inference.py \
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
        SUBMISSION_PATH="../outputs/${DATASET}/${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"
        KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"
        DATA_PATH="../../data/${DATASET}/archehr-qa.xml"
        OUT_FILE_PATH="../results/${DATASET}/${MODEL_NAME}_prompt_${PROMPT_INDEX}.json"
        mkdir -p "../results/${DATASET}"
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
