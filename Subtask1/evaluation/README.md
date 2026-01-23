# Evaluation

This directory contains the scoring scripts for the ArchEHR-QA 2026 Shared Task.

## Setup

The scripts are tested using Python 3.8.18.

### 1. Install the required dependencies.

```bash
$ pip install --no-cache-dir -r requirements.txt
```

### 2. (Required to run the Subtask 1 and 3 scoring scripts locally) QuickUMLS setup.

Generate the QuickUMLS data files following the directions provided at https://github.com/Georgetown-IR-Lab/QuickUMLS (requires a UMLS license).

> Provide the QuickUMLS data directory to the scripts via `--quickumls_path`.

### 3. Run the scoring scripts.

#### Subtask 1

```bash
python scoring_subtask_1.py \
    --submission_path submission.json \
    --key_path archehr-qa.xml \
    --quickumls_path quickumls/ \
    --out_file_path scores.json
```

If you have any questions, please reach out to sarvesh.soni@nih.gov.