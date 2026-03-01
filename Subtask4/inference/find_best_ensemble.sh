#!/bin/bash
#SBATCH --job-name=ensemble_search
#SBATCH --output=../logs/ensemble_search_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "========================================"

source .venv/bin/activate

cd /ceph/home/student.aau.dk/ge48ab/ArchEHR-QA-2026/Subtask4/inference

uv run python find_best_ensemble.py

echo "========================================"
echo "Finished at: $(date)"