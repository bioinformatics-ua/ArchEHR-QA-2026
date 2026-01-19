#!/bin/bash
#SBATCH --job-name=archehr_audit     # Job name
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks-per-node=1         # 1 main task per array index
#SBATCH --gpus-per-task=l40s:1      # 1 L40S GPU per array index
#SBATCH --cpus-per-task=8           # 8 CPU cores per GPU
#SBATCH --mem-per-gpu=64G           # Memory allocation
#SBATCH --output=logs/%x_%A_%a.out  # Logs sorted by JobID and ArrayID
#SBATCH --array=0 3                 # Creates 4 tasks (IDs 0, 1, 2, 3)

# --- Configuration ---
WORLD_SIZE=4                        # Must match total array count (0-3)
# MODEL_ID="gaunernst/gemma-3-27b-it-int4-awq"
MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

BASE_DIR="./dataset_1.3/dev/"
OUTPUT_FILE="results/audit_results.jsonl"

echo "--- Starting Audit Task $SLURM_ARRAY_TASK_ID of $WORLD_SIZE ---"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
source .venv/bin/activate

# --- Run Audit Script ---
# We pass the SLURM_ARRAY_TASK_ID as the 'rank'
python3 run.py \
    --model-id "$MODEL_ID" \
    --rank $SLURM_ARRAY_TASK_ID \
    --world-size $WORLD_SIZE \
    --base-dir "$BASE_DIR" \
    --output-file "$OUTPUT_FILE"

echo "--- Task $SLURM_ARRAY_TASK_ID Finished ---"