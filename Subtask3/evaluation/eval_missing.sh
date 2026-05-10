#!/bin/bash
#SBATCH --job-name=subtask3_eval_missing
#SBATCH --output=../logs/eval_missing_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
SUBTASK_DIR="${ROOT_DIR}/Subtask3"

source "${ROOT_DIR}/.venv/bin/activate"

SIF_IMAGE="./builder.sif"
UV_BIN=$(which uv)

# Model lists
open_source_models=(
    'meta-llama-Llama-3-1-8B-Instruct'

)
closed_source_models=(
    'anthropic-claude-sonnet-4-5'

)

# Directories
DATASET="test"
DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"
INPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"
ANALYSIS_DIR="${SUBTASK_DIR}/analysis/${DATASET}"
LOGS_DIR="${SUBTASK_DIR}/logs"
KEY_PATH="${DATA_DIR}/archehr-qa_key.json"
DATA_PATH="${DATA_DIR}/archehr-qa.xml"
QUICKUMLS_PATH="./quickumls/final"

mkdir -p "$RESULTS_DIR" "$ANALYSIS_DIR" "$LOGS_DIR"

echo "Starting Evaluation for Missing Results..."

# Loop over all models and prompts
for model in "${open_source_models[@]}" "${closed_source_models[@]}"; do
    for prompt in {1..11}; do
        model_name="$model"
        output_json="${INPUT_DIR}/${model_name}_prompt_${prompt}.json"
        result_json="${RESULTS_DIR}/${model_name}_prompt_${prompt}.json"
        if [ -f "$output_json" ]; then
            if [ -f "$result_json" ]; then
                echo "[SKIP] Result exists for: $model_name (prompt $prompt)"
            else
                echo "[EVAL] $model_name (prompt $prompt)"
                singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python scoring_subtask_3.py \
                    --submission_path "$output_json" \
                    --key_path "$KEY_PATH" \
                    --data_path "$DATA_PATH" \
                    --quickumls_path "$QUICKUMLS_PATH" \
                    --out_file_path "$result_json"
                echo "[DONE] $model_name (prompt $prompt)"
            fi
        else
            echo "[MISSING] Output not found for: $model_name (prompt $prompt)"
        fi
    done
done

echo "All missing evaluations complete."
