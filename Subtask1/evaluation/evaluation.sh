#!/bin/bash

#SBATCH --job-name=subtask1_evaluation
#SBATCH --output=../logs/eval_batch_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail
shopt -s nullglob

# Run this script from its own directory so relative paths are stable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ================= CONFIGURATION =================
# RUN_MODE=local runs evaluations in this process.
# RUN_MODE=slurm submits one SLURM job per missing result file.
RUN_MODE="${RUN_MODE:-local}"                 # local or slurm
DATASET="${DATASET:-new}"                     # dev, test, or new

DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../data}"
INPUT_ROOT="${INPUT_ROOT:-../outputs_v2}"
RESULT_ROOT="${RESULT_ROOT:-../results_v2}"
QUICKUMLS_PATH="${QUICKUMLS_PATH:-../quickumls/final}"
LOG_DIR="${LOG_DIR:-../logs}"
SIF_IMAGE="${SIF_IMAGE:-./builder.sif}"
UV_BIN="${UV_BIN:-$(which uv)}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
GPU_COUNT="${GPU_COUNT:-1}"
SUBMIT_DELAY_SECONDS="${SUBMIT_DELAY_SECONDS:-0.5}"

INPUT_DIR="${INPUT_ROOT}/${DATASET}"
RESULTS_DIR="${RESULT_ROOT}/${DATASET}"
KEY_PATH="${DATA_ROOT}/${DATASET}/archehr-qa.xml"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "========================================"
echo "Starting Evaluation"
echo "  mode:          $RUN_MODE"
echo "  dataset:       $DATASET"
echo "  input dir:     $(realpath -m "$INPUT_DIR")"
echo "  results dir:   $(realpath -m "$RESULTS_DIR")"
echo "  key path:      $(realpath -m "$KEY_PATH")"
echo "  quickumls:     $(realpath -m "$QUICKUMLS_PATH")"
echo "========================================"

submission_paths=("$INPUT_DIR"/*.json)
if [ ${#submission_paths[@]} -eq 0 ]; then
    echo "No submission JSON files found in $INPUT_DIR"
    exit 0
fi

run_evaluation() {
    local submission_path="$1"
    local out_file_path="$2"

    singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python evaluation.py \
        --submission_path "$submission_path" \
        --key_path "$KEY_PATH" \
        --quickumls_path "$QUICKUMLS_PATH" \
        --out_file_path "$out_file_path"
}

submit_evaluation() {
    local submission_path="$1"
    local out_file_path="$2"
    local model_name="$3"
    local job_script="temp_eval_${model_name}.slurm"

    cat <<EOF > "$job_script"
#!/bin/bash

#SBATCH --job-name=eval_${model_name}
#SBATCH --output=${LOG_DIR}/eval_${model_name}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=gpu:${GPU_COUNT}

set -euo pipefail

echo "Job started on \$(hostname)"
echo "Job ID: \${SLURM_JOB_ID:-manual}"

cd "$SCRIPT_DIR"

singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python evaluation.py \\
    --submission_path "$submission_path" \\
    --key_path "$KEY_PATH" \\
    --quickumls_path "$QUICKUMLS_PATH" \\
    --out_file_path "$out_file_path"

echo "[DONE] Finished: ${model_name}"
EOF

    sbatch "$job_script"
    rm "$job_script"
}

for submission_path in "${submission_paths[@]}"; do
    filename="$(basename -- "$submission_path")"
    model_name="${filename%.*}"
    out_file_path="${RESULTS_DIR}/${model_name}.json"

    if [ -f "$out_file_path" ]; then
        echo "[SKIP] Result already exists for: $model_name"
        continue
    fi

    case "$RUN_MODE" in
        local)
            echo "[RUNNING] Processing: $model_name"
            run_evaluation "$submission_path" "$out_file_path"
            echo "[DONE] Finished: $model_name"
            ;;
        slurm)
            echo "[SUBMIT] Launching evaluation for: $model_name"
            submit_evaluation "$submission_path" "$out_file_path" "$model_name"
            sleep "$SUBMIT_DELAY_SECONDS"
            ;;
        *)
            echo "Error: RUN_MODE must be 'local' or 'slurm'." >&2
            exit 1
            ;;
    esac
done

echo "========================================"
echo "All evaluation checks complete."
