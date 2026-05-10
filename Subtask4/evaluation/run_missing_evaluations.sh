#!/bin/bash
#SBATCH --job-name=subtask4_eval_missing
#SBATCH --output=../logs/eval_missing_subtask4_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo "Running missing evaluations for Subtask 4"
echo "Target models:"
echo "  google/medgemma-4b-it"
echo "  google/gemma-3-27b-it"
echo "  google/medgemma-1.5-4b-it"
echo "  khazarai/Bio-8B-it"
echo "  Qwen/Qwen3-8B"
echo "  Qwen/Qwen3-32b"
echo "  BioMistral/BioMistral-7B"
echo "  Echelon-AI/Med-Qwen2-7B"
echo "  meta-llama/Llama-3.1-8B-Instruct"
echo "========================================"

# Activate environment
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT_DIR="$(cd "${RUN_DIR}/../.." && pwd)"
SUBTASK_DIR="${ROOT_DIR}/Subtask4"

source "${ROOT_DIR}/.venv/bin/activate"
mkdir -p "${SUBTASK_DIR}/logs"

# Run the missing evaluations script
uv run python run_missing_evaluations.py

echo "[DONE] Missing evaluations complete."
