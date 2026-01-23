#!/bin/bash

#SBATCH --job-name=subtask1_inference
#SBATCH --output=../logs/inference%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"


source .venv/bin/activate

# --- Run the Inference Script ---
echo "Starting Python inference script..."
python inference.py \
    --xml-file ../../data/dev/archehr-qa.xml \
    --prompt-file prompt.json \
    --prompt-index 1 \
    --output-file ../outputs/predictions.json \
    # --model "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8" \ 
    # --mode "train/test"


# {"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8" : "nemotron3"}


# nemotron3_prompt_1.json 



# # --- Run Evaluation Script ---
# echo "Starting evaluation script..."
# cd ../evaluation
# uv run python evaluation.py \
#     --submission_path ../outputs/predictions.json \
#     --key_path ../../data/dev/archehr-qa.xml \
#     --quickumls_path ../quickumls/ \
#     --out_file_path ../results/nemotron3_prompt_1_results.json


# echo "Evaluation complete. Job finished."
