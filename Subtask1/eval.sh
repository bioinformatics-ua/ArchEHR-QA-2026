#!/bin/bash

#SBATCH --job-name=vllm-batch-inference  # A descriptive name for your job
#SBATCH --output=logs/eval%j.out         # Standard output and error log (%j is the job ID)
#SBATCH --nodes=1                        # We need just one machine
#SBATCH --ntasks=1                       # One task (our Python script)
#SBATCH --cpus-per-task=4                # Request 8 CPU cores
#SBATCH --mem=16                        # Request 64GB of RAM
#SBATCH --gres=gpu:1                     # IMPORTANT: Request 1 GPU
#SBATCH --time=00:30:00                  # Maximum job run time (HH:MM:SS)

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate your Python virtual environment created with uv
# This is crucial to access your installed packages like vllm
source venv-eval/bin/activate


# --- Run the Evaluation Script ---

python eval-script.py outputs/gold.json
