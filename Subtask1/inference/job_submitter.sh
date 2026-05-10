#!/bin/bash

set -euo pipefail

# =============================================================================
# STABLE SCRIPT LOCATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT_DIR="$(git rev-parse --show-toplevel)"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Example:
# MODE=cloud DATASET=test PROMPT_START=1 PROMPT_END=10 bash job_submitter.sh

MODE="${MODE:-local}"                 # local or cloud
DATASET="${DATASET:-dev}"             # dev or test
PROMPT_START="${PROMPT_START:-1}"
PROMPT_END="${PROMPT_END:-10}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/Subtask1/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/Subtask1/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/Subtask1/results}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/Subtask1/logs}"

RUN_EVAL="${RUN_EVAL:-true}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SUBMIT_DELAY_SECONDS="${SUBMIT_DELAY_SECONDS:-0.5}"

TEMP_JOB_DIR="./temp_jobs"

mkdir -p "$TEMP_JOB_DIR"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
LOCAL_MODELS=(
    "Echelon-AI/Med-Qwen2-7B"
    "Qwen/Qwen3-32B"
    "google/gemma-3-27b-it"
    "google/medgemma-1.5-4b-it"
    "google/medgemma-27b-text-it"
    "mistralai/Ministral-3-14B-Reasoning-2512"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
)

CLOUD_MODELS=(
    "anthropic/claude-sonnet-4.5"
    "openai/gpt-5.2"
    "google/gemini-3-flash-preview"
    "qwen/qwen3-max"
)

case "$MODE" in
    local)
        MODELS=("${LOCAL_MODELS[@]}")
        GPU_COUNT="${GPU_COUNT:-4}"
        ;;
    cloud)
        MODELS=("${CLOUD_MODELS[@]}")
        GPU_COUNT="${GPU_COUNT:-1}"
        ;;
    *)
        echo "Error: MODE must be 'local' or 'cloud'."
        exit 1
        ;;
esac

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"
RESULT_DIR="${RESULT_ROOT}/${DATASET}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

# =============================================================================
# INFO
# =============================================================================
echo "========================================"
echo "Batch submitter configuration"
echo "  mode:         $MODE"
echo "  dataset:      $DATASET"
echo "  prompts:      ${PROMPT_START}-${PROMPT_END}"
echo "  gpu count:    $GPU_COUNT"
echo "  output dir:   $(realpath -m "$OUTPUT_DIR")"
echo "========================================"

# =============================================================================
# SUBMIT JOBS
# =============================================================================
for MODEL in "${MODELS[@]}"; do
    for PROMPT_INDEX in $(seq "$PROMPT_START" "$PROMPT_END"); do

        MODEL_NAME_CLEAN="$(echo "$MODEL" | tr '/' '-' | tr '.' '-')"

        OUTPUT_FILE="${MODEL_NAME_CLEAN}_prompt_${PROMPT_INDEX}.json"

        OUTPUT_PATH="$(realpath -m "${OUTPUT_DIR}/${OUTPUT_FILE}")"

        RESULT_PATH="$(realpath -m "${RESULT_DIR}/${OUTPUT_FILE}")"

        XML_PATH="$(realpath -m "${DATA_ROOT}/${DATASET}/archehr-qa.xml")"

        if [ -f "$OUTPUT_PATH" ]; then
            echo "[SKIP] Exists: $OUTPUT_FILE"
            continue
        fi

        echo "[SUBMIT] $OUTPUT_FILE"

        JOB_SCRIPT="${TEMP_JOB_DIR}/temp_job_${MODEL_NAME_CLEAN}_${PROMPT_INDEX}.slurm"

        cat <<EOF > "$JOB_SCRIPT"
#!/bin/bash

#SBATCH --job-name=inf_${MODEL_NAME_CLEAN:0:10}_${PROMPT_INDEX}
#SBATCH --output=${LOG_DIR}/inference_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=gpu:${GPU_COUNT}

set -euo pipefail

echo "Job started on \$(hostname)"
echo "Job ID: \${SLURM_JOB_ID:-manual}"

cd "$SCRIPT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${ROOT_DIR}/common:\${PYTHONPATH:-}"

MODE="$MODE"
MODEL="$MODEL"
PROMPT_INDEX="$PROMPT_INDEX"

OUTPUT_PATH="$OUTPUT_PATH"
RESULT_PATH="$RESULT_PATH"
XML_PATH="$XML_PATH"
KEY_PATH="$(realpath -m "${DATA_ROOT}/${DATASET}/archehr-qa_key.json")"

RUN_EVAL="$RUN_EVAL"

if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

echo "Running inference"
echo "  model:  \$MODEL"
echo "  prompt: \$PROMPT_INDEX"

PYTHONUNBUFFERED=1 python inference.py \\
    --xml-file "\$XML_PATH" \\
    --prompt-file prompt.json \\
    --prompt-index "\$PROMPT_INDEX" \\
    --output-file "\$OUTPUT_PATH" \\
    --inference-mode "$MODE" \\
    --model "\$MODEL"

if [ "\$RUN_EVAL" = "true" ]; then

    echo "Running evaluation"

    cd ../evaluation

    SIF_IMAGE="./builder.sif"
    UV_BIN="\$(which uv)"

    singularity exec --nv "\$SIF_IMAGE" "\$UV_BIN" run python evaluation.py \\
        --submission_path "\$OUTPUT_PATH" \\
        --key_path "\$KEY_PATH" \\
        --quickumls_path ../quickumls/final \\
        --out_file_path "\$RESULT_PATH"
fi

echo "Job complete."

EOF

        sbatch "$JOB_SCRIPT"

        rm "$JOB_SCRIPT"

        sleep "$SUBMIT_DELAY_SECONDS"

    done
done

echo "========================================"
echo "All jobs submitted."