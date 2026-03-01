#!/bin/bash
#SBATCH --job-name=subtask4_dev
#SBATCH --output=../logs/dev_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "========================================"

cd /ceph/home/student.aau.dk/vf36ha/ArchEHR-QA-2026/Subtask4/inference

/ceph/home/student.aau.dk/vf36ha/.local/bin/uv run python run_dev_pipeline.py

echo "Finished at: $(date)"