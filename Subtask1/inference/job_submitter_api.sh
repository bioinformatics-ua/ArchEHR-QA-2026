#!/bin/bash

# ================= CONFIGURATION =================
DATASET="new" # Change to "test" if needed
LOG_DIR="../logs"
OUTPUT_DIR="../outputs_v2/${DATASET}"


MODELS=(
    #     "moonshotai/kimi-k2.5"
    #     "z-ai/glm-4.7"
    "anthropic/claude-sonnet-4.5"
    "openai/gpt-5.2"
    "google/gemini-3-flash-preview"
    "qwen/qwen3-max"
#     "openai/gpt-oss-120b"
    # "qwen/qwen3-235b-a22b-2507"
    # "deepseek/deepseek-r1-0528:free"
    # "google/gemini-3-pro-preview"
)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ================= MAIN LOOP =================

for MODEL in "${MODELS[@]}"; do
    for PROMPT_INDEX in {1..10}; do

        # 1. Calculate the expected filename
        # Logic: Replace / and . with - 
        MODEL_NAME_CLEAN=$(echo "$MODEL" | tr '/' '-' | tr '.' '-')
        OUTPUT_FILE="${MODEL_NAME_CLEAN}_prompt_${PROMPT_INDEX}.json"
        FULL_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"

        # 2. Check if file exists
        if [ -f "$FULL_PATH" ]; then
            echo "✅ [SKIP] Exists: $OUTPUT_FILE"
        else
            echo "🚀 [SUBMIT] Missing: $OUTPUT_FILE (Model: $MODEL | Prompt: $PROMPT_INDEX)"

            # 3. Create a temporary SLURM script for this specific job
            # We use 'EOF' to allow variable expansion for $MODEL and $PROMPT_INDEX,
            # but we escape internal bash variables (like \$SLURM_JOB_ID) so they evaluate later.
            
            JOB_SCRIPT="temp_job_${MODEL_NAME_CLEAN}_${PROMPT_INDEX}.slurm"

            cat <<EOF > "$JOB_SCRIPT"
#!/bin/bash

#SBATCH --job-name=inf_${MODEL_NAME_CLEAN:0:10}_${PROMPT_INDEX}
#SBATCH --output=${LOG_DIR}/inference_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
echo "Job started on \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"

source .venv/bin/activate

# =============================================================================
# INJECTED VARIABLES FROM SUBMITTER SCRIPT
# =============================================================================
MODE="cloud"
MODEL="${MODEL}"
DATASET="${DATASET}"
PROMPT_INDEX=${PROMPT_INDEX}

# Auto-generate output filename inside the job (keeping consistency)
MODEL_NAME=\$(echo "\$MODEL" | tr '/' '-' | tr '.' '-')
OUTPUT_FILE="\${MODEL_NAME}_prompt_\${PROMPT_INDEX}.json"

# Load .env file
if [ -f .env ]; then
    export \$(cat .env | xargs)
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found"
fi

# --- Run the Inference Script ---
echo "Starting Python inference script for ${MODEL}..."
PYTHONUNBUFFERED=1 python inference.py \\
    --xml-file ../../data/${DATASET}/archehr-qa.xml \\
    --prompt-file prompt.json \\
    --prompt-index \$PROMPT_INDEX \\
    --output-file ../outputs_v2/${DATASET}/\$OUTPUT_FILE \\
    --inference-mode "\$MODE" \\
    --model "\$MODEL"

# Deactivate inference venv
deactivate

cd ../evaluation 
SIF_IMAGE="./builder.sif"
UV_BIN=\$(which uv)

# Run Evaluation
singularity exec --nv "\$SIF_IMAGE" "\$UV_BIN" run python evaluation.py \\
    --submission_path ../outputs/${DATASET}/\$OUTPUT_FILE \\
    --key_path ../../data/${DATASET}/archehr-qa.xml \\
    --quickumls_path ../quickumls/final \\
    --out_file_path ../results_v2/${DATASET}/\$OUTPUT_FILE

EOF

            # 4. Submit the generated script and then delete it
            sbatch "$JOB_SCRIPT"
            rm "$JOB_SCRIPT"
            
            # Optional: Sleep briefly to prevent overloading the scheduler
            sleep 0.5
        fi
    done
done

echo "========================================"
echo "All checks complete."