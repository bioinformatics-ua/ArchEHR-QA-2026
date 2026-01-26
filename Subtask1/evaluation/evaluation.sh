#!/bin/bash

#SBATCH --job-name=subtask1_evaluation
#SBATCH --output=../logs/eval_batch_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
SIF_IMAGE="./builder.sif"
UV_BIN=$(which uv)

# Define directories
INPUT_DIR="../outputs/dev"
RESULTS_DIR="../results/dev"
# singularity exec --nv "$SIF_IMAGE" "$UV_BIN" sync

# Ensure results directory exists

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
        singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python evaluation.py \
            --submission_path "$submission_path" \
            --key_path ../../data/dev/archehr-qa.xml \
            --quickumls_path ../quickumls/final \
            --out_file_path "$out_file_path"
            
        echo "[DONE] Finished: $model_name"
    fi

done

echo "All evaluations complete."