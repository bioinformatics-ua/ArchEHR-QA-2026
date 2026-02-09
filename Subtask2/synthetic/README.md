# Synthetic Sentence Label Generation

This module generates synthetic sentence-level relevance labels for ArchEHR-QA test cases using Large Language Models.

## Overview

The script classifies each sentence in clinical notes as:
- **essential**: Critical information directly needed to answer the question
- **supplementary**: Supporting context or additional relevant details  
- **not-relevant**: Does not contribute to answering the question

## Requirements

The script uses the `common` package providers for LLM inference. Install dependencies:

```bash
cd ../../common
pip install -e .
cd ../Subtask2/synthetic
pip install -r requirements.txt
```

## Usage

### Basic Usage (Cloud-based)

```bash
python generate_labels.py \
  --xml-file ../../data/test_2025/archehr-qa.xml \
  --output-file ../../data/test_2025/archehr-qa_synthetic_labels.json \
  --inference-mode cloud \
  --model anthropic/claude-3.5-sonnet
```

### Local Model Usage

```bash
python generate_labels.py \
  --xml-file ../../data/test_2025/archehr-qa.xml \
  --output-file ../../data/test_2025/archehr-qa_synthetic_labels.json \
  --inference-mode local \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Resume from Checkpoint

If the script is interrupted, you can resume:

```bash
python generate_labels.py \
  --xml-file ../../data/test_2025/archehr-qa.xml \
  --output-file ../../data/test_2025/archehr-qa_synthetic_labels.json \
  --inference-mode cloud \
  --model anthropic/claude-3.5-sonnet \
  --resume-from 50
```

## Script Options

- `--xml-file`: Path to input XML file with cases
- `--output-file`: Path to save synthetic labels (JSON format)
- `--inference-mode`: `cloud` for OpenRouter API or `local` for vLLM
- `--model`: Model identifier (e.g., `anthropic/claude-3.5-sonnet` for cloud, HuggingFace model path for local)
- `--batch-size`: Number of cases to process in parallel (default: 1)
- `--resume-from`: Resume processing from case index (default: 0)

## Environment Variables

### For Cloud Inference
```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

### For Local Inference
```bash
export HF_TOKEN="your-huggingface-token"  # Optional, for gated models
```

## Output Format

The script generates a JSON file matching the format of `archehr-qa_key.json`:

```json
[
    {
        "case_id": "1",
        "answers": [
            {
                "sentence_id": "1",
                "relevance": "not-relevant"
            },
            {
                "sentence_id": "2",
                "relevance": "essential"
            },
            ...
        ]
    },
    ...
]
```

## Shell Script

For convenience, use the provided shell script:

```bash
./run_generation.sh
```

Edit the script to customize parameters.

## Notes

- The script saves intermediate results after each batch
- Use resume functionality for long-running tasks
- Cloud inference is recommended for better quality
- Adjust batch size based on available resources
