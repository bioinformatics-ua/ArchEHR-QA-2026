#!/bin/bash
#SBATCH --job-name=ensemble_search
#SBATCH --output=../logs/ensemble_search_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

RUN_ID=${1:-default}

echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "Started at: $(date)"
echo "========================================"

source .venv/bin/activate

cd /ceph/home/student.aau.dk/ge48ab/ArchEHR-QA-2026/Subtask4/inference

uv run python find_best_ensemble.py $RUN_ID

echo "========================================"
echo "Finished at: $(date)"