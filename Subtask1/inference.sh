#!/bin/bash

#SBATCH --job-name=vllm-batch-inference  # A descriptive name for your job
#SBATCH --output=logs/inference%j.out    # Standard output and error log (%j is the job ID)
#SBATCH --nodes=1                        # We need just one machine
#SBATCH --ntasks=1                       # One task (our Python script)
#SBATCH --cpus-per-task=4                # Request 4 CPU cores
#SBATCH --mem=16G                        # Request 16GB of RAM
#SBATCH --gres=gpu:1                     # IMPORTANT: Request 1 GPU
#SBATCH --time=01:00:00                  # Maximum job run time (HH:MM:SS)

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate the unified virtual environment from repo root
# This contains vllm, torch, transformers, accelerate
source ../.venv/bin/activate
echo "Virtual environment activated."

# Optional: Set Hugging Face token if you are using a gated model
# export HF_TOKEN="your_hugging_face_token_here"

# --- Run the Inference Script ---
echo "Starting Python inference script..."
python run_inference.py \
    --xml-file ../data/dev/archehr-qa.xml \
    --prompt-file prompt.jsonl \
    --output-file predictions.json

echo "Inference complete. Deactivating inference environment..."
deactivate

# --- Switch to Eval Environment ---
echo "Activating eval environment..."
source venv-eval/bin/activate
echo "Eval environment activated."

# --- Run Evaluation Script ---
echo "Starting evaluation script..."
python eval-script.py outputs/predictions.json

echo "Evaluation complete. Job finished."