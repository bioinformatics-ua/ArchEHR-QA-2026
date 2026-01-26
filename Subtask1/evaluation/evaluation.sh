#!/bin/bash

#SBATCH --job-name=subtask1_evaluation
#SBATCH --output=../logs/eval%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
SIF_IMAGE="./builder.sif"
UV_BIN=$(which uv)

# 1. Build dependencies (Fixes the C++ error)
# singularity exec --nv "$SIF_IMAGE" "$UV_BIN" sync

# 2. Run the code
singularity exec --nv "$SIF_IMAGE" "$UV_BIN" run python evaluation.py \
    --submission_path ../outputs/dev/google-gemini-3-pro-preview_prompt_3.json\
    --key_path ../../data/dev/archehr-qa.xml \
    --quickumls_path ../quickumls/final \
    --out_file_path ../results/dev/google-gemini-3-pro-preview_prompt_3.json\