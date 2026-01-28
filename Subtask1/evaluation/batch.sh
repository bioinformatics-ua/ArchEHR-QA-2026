#!/bin/bash

# --- Configuration ---
INPUT_DIR="../outputs_v2/new"
RESULTS_DIR="../results_v2/new"
LOG_DIR="../logs"

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Starting Evaluation Submitter"
echo "Input: $INPUT_DIR"
echo "Output: $RESULTS_DIR"
echo "========================================"

# --- The Loop ---
for submission_path in "$INPUT_DIR"/*.json; do
    
    # 1. Parse the Model Name
    filename=$(basename -- "$submission_path")
    model_name="${filename%.*}"
    
    # Define expected output path
    out_file_path="$RESULTS_DIR/${model_name}.json"

    # 2. Check if result already exists
    if [ -f "$out_file_path" ]; then
        echo "✅ [SKIP] Result already exists for: $model_name"
    else
        echo "🚀 [SUBMIT] Launching evaluation for: $model_name"
        
        # 3. Create a temporary SLURM script for this specific job
        # We use a unique temporary filename based on the model name
        JOB_SCRIPT="temp_eval_${model_name}.slurm"

        # Start Heredoc to write the SLURM script
        # Note: Variables like ${model_name} are expanded NOW (by the submitter).
        # Variables like \$SLURM_JOB_ID or \$(which uv) are escaped so they run LATER (on the node).
        cat <<EOF > "$JOB_SCRIPT"
#!/bin/bash

#SBATCH --job-name=eval_${model_name}
#SBATCH --output=${LOG_DIR}/eval_${model_name}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
echo "Job started on \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"

# Define local paths inside the compute node
SIF_IMAGE="./builder.sif"
UV_BIN=\$(which uv)

echo "Processing Model: ${model_name}"

# --- Run the Evaluation ---
# We use backslashes (\\) at end of lines for python args to keep the generated file clean
singularity exec --nv "\$SIF_IMAGE" "\$UV_BIN" run python evaluation.py \\
    --submission_path "${submission_path}" \\
    --key_path ../../data/new/archehr-qa.xml \\
    --quickumls_path ../quickumls/final \\
    --out_file_path "${out_file_path}"

echo "[DONE] Finished: ${model_name}"

EOF

        # 4. Submit the generated script and then delete it
        sbatch "$JOB_SCRIPT"
        rm "$JOB_SCRIPT"

        # Optional: Sleep briefly to prevent overloading the scheduler if you have hundreds of files
        sleep 0.5
    fi

done

echo "========================================"
echo "All submission checks complete."