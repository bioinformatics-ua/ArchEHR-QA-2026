#!/bin/bash
#SBATCH --job-name=subtask3_open_batch
#SBATCH --output=../logs/open_batch_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

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


# List of open source models (exact names)
OPEN_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct" #
    "google/medgemma-27b-text-it" #
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-32B"
    "Qwen/Qwen3-8B"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    "google/medgemma-4b-it"
    "Echelon-AI/Med-Qwen2-7B" #
    "google/medgemma-1.5-4b-it" #
    "BioMistral/BioMistral-7B"
    "khazarai/Bio-8B-it"
)

source .venv/bin/activate

# Load .env for HF_TOKEN / OPENROUTER_API_KEY
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# Accept model index (1-based) as argument
MODEL_INDEX="$1"
if [[ -z "$MODEL_INDEX" ]]; then
    echo "Usage: sbatch open_batch.sh <model_index>"
    exit 1
fi
if ! [[ "$MODEL_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Model index must be a number."
    exit 1
fi
if (( MODEL_INDEX < 1 || MODEL_INDEX > ${#OPEN_MODELS[@]} )); then
    echo "Model index out of range."
    exit 1
fi
MODEL="${OPEN_MODELS[$((MODEL_INDEX-1))]}"
echo "Selected model: $MODEL"

for PROMPT_INDEX in {1..11}; do
    MODEL_NAME=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
    echo "Processing $MODEL (prompt $PROMPT_INDEX)"
    uv run python inference.py \
        --xml-file "${DATA_DIR}/archehr-qa.xml" \
        --prompt-file "$PROMPT_FILE" \
        --prompt-index $PROMPT_INDEX \
        --output-file "${OUTPUT_DIR}/${MODEL_NAME}_prompt_${PROMPT_INDEX}.json" \
        --inference-mode "local" \
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

