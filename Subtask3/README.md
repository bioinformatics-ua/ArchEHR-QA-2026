# Subtask 3

This subdirectory contains the scripts and instructions for Subtask 3 of the ArchEHR-QA-2026 Shared Task.

Given a patient narrative, a clinician question, and a clinical note, the task is to generate a concise evidence-grounded answer (max 75 words).

It uses `uv` to manage the Python environment and dependencies. You can install it from [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Subtask 3 supports both cloud-based inference (via OpenRouter) and local inference (via vLLM), with SLURM scripts provided to facilitate running on an HPC cluster.

## Data

The `data/` directory is not included in the repository and must be created manually. Subtask 3 uses three splits: `dev` and `test` (both with ground-truth keys, used for inference and evaluation) and `test-2026` (blind test set, inference only).

```
data/
├── dev/                        # Development set — inference + evaluation
│   ├── archehr-qa.xml
│   └── archehr-qa_key.json
├── test/                       # Test set — inference + evaluation
│   ├── archehr-qa.xml
│   └── archehr-qa_key.json
└── test-2026/                  # Blind test set — inference only (no key)
    └── archehr-qa.xml
```

The data files can be obtained from the official ArchEHR-QA-2026 shared task.

## Inference

Supports local (vLLM) and cloud (OpenRouter) inference, as well as batch sweeps and ensemble judging. Please refer to the [Inference README](./inference/README.md) for detailed instructions.

## Evaluation

Uses the official ArchEHR-QA-2026 evaluation script via a Singularity container with QuickUMLS. Please refer to the [Evaluation README](./evaluation/README.md) for detailed instructions.

## Other directories

| Directory    | Description                              |
| ------------ | ---------------------------------------- |
| `logs/`      | Directory to store SLURM output files.   |
| `outputs/`   | Directory to store model predictions.    |
| `results/`   | Directory to store evaluation results.   |
| `analysis/`  | Directory for analysis scripts and notebooks. |
| `data/`      | Directory to store task data files.      |
