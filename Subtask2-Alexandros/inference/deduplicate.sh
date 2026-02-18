#!/bin/bash
#SBATCH --job-name=subtask2_dedup
#SBATCH --output=../logs/dedup_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET="dev"               # dev / test / test-2026

# Input: the predictions file you want to deduplicate
INPUT_FILE="../outputs/${DATASET}/google-gemini-2-5-flash_prompt_9.json"

# Similarity threshold: Zone-B sentence is dropped if its Jaccard similarity
# with any Zone-A sentence in the prediction is >= this value.
# 0.2 = aggressive (removes loose paraphrases)
# 0.3 = moderate (removes content-word overlap, default)
# 0.5 = conservative (only removes near-verbatim copies)
SIMILARITY_THRESHOLD=0.7

# Auto-generate output filename
INPUT_BASENAME=$(basename "$INPUT_FILE" .json)
THRESHOLD_TAG=$(echo "$SIMILARITY_THRESHOLD" | tr '.' '-')
OUTPUT_FILE="../outputs/${DATASET}/${INPUT_BASENAME}_dedup${THRESHOLD_TAG}.json"

# =============================================================================
# DEDUPLICATION
# =============================================================================
echo "[1/3] Running section-aware deduplication..."

uv run python deduplicate.py \
    --input-file  "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --xml-file    "../../data/${DATASET}/archehr-qa.xml" \
    --threshold   $SIMILARITY_THRESHOLD

echo "[1/3] Deduplication complete."

# =============================================================================
# EVALUATION
# =============================================================================
KEY_PATH="../../data/${DATASET}/archehr-qa_key.json"

if [ ! -f "${KEY_PATH}" ]; then
    echo "[2/3] No key file found for ${DATASET}, skipping evaluation."
    echo "========================================"
    echo "Output: ${OUTPUT_FILE}"
    exit 0
fi

echo "[2/3] Running evaluation..."

deactivate

cd ../evaluation
source .venv/bin/activate

OUTPUT_BASENAME=$(basename "$OUTPUT_FILE")
SUBMISSION_PATH="../outputs/${DATASET}/${OUTPUT_BASENAME}"
OUT_FILE_PATH="../results/${DATASET}/${OUTPUT_BASENAME}"

mkdir -p "../results/${DATASET}"

uv run python scoring_subtask_2.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --out_file_path "$OUT_FILE_PATH"

echo "[2/3] Evaluation complete."

# =============================================================================
# RESULTS ANALYSIS (per-sentence P/R/F1)
# =============================================================================
echo "[3/3] Running per-sentence analysis..."

ANALYSIS_OUT_PATH="../results-analysis/${DATASET}/${OUTPUT_BASENAME}"
mkdir -p "../results-analysis/${DATASET}"

uv run python results_analysis.py \
    --submission_path "$SUBMISSION_PATH" \
    --key_path "$KEY_PATH" \
    --xml_path "../../data/${DATASET}/archehr-qa.xml" \
    --out_file_path "$ANALYSIS_OUT_PATH"

echo "[3/3] Analysis complete."
echo "========================================"
echo "Results      : ${OUT_FILE_PATH}"
echo "Per-sentence : ${ANALYSIS_OUT_PATH}"
