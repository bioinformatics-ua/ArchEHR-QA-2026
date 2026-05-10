#!/bin/bash
#SBATCH --job-name=subtask3_evaluation
#SBATCH --output=../logs/eval_batch_%j.out
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

# Define directories
DATASET="dev"
DATA_DIR="${SUBTASK_DIR}/data/${DATASET}"
INPUT_DIR="${SUBTASK_DIR}/outputs/${DATASET}"
RESULTS_DIR="${SUBTASK_DIR}/results/${DATASET}"
ANALYSIS_DIR="${SUBTASK_DIR}/analysis/${DATASET}"
LOGS_DIR="${SUBTASK_DIR}/logs"
# singularity exec --nv "$SIF_IMAGE" "$UV_BIN" sync

# Ensure results directory exists
mkdir -p "$RESULTS_DIR" "$ANALYSIS_DIR" "$LOGS_DIR"

echo "Starting Batch Evaluation..."

# --- The Loop ---
for submission_path in "$INPUT_DIR"/*.json; do
    
    # 1. Parse the Model Name
    filename=$(basename -- "$submission_path")
    model_name="${filename%.*}"
    
    # Define expected output path
    out_file_path="$RESULTS_DIR/${model_name}.json"

    # 2. Check if result already exists
    if [ -f "$out_file_path" ]; then
        echo "[SKIP] Result already exists for: $model_name"
    else
        echo "[RUNNING] processing: $model_name"
        
        # 3. Run the code
        singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python scoring_subtask_3.py \
        --submission_path "$submission_path" \
        --key_path "${DATA_DIR}/archehr-qa_key.json" \
        --data_path "${DATA_DIR}/archehr-qa.xml" \
        --quickumls_path ./quickumls/final \
        --out_file_path "$out_file_path" \
        --case_ids_to_score 1-3
            
        echo "[DONE] Finished: $model_name"
    fi

done

echo "All evaluations complete."