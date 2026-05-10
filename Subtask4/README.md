# Subtask 4

This subdirectory contains the scripts and instructions for Subtask 4 of the ArchEHR-QA-2026 Shared Task.

Given a clinician's answer and a clinical note, the task is to map each answer sentence to the note sentence(s) that support it.

It uses `uv` to manage the Python environment and dependencies. You can install it from [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Subtask 4 supports both cloud-based inference (via OpenRouter) and local inference (via vLLM), with SLURM scripts provided to facilitate running on an HPC cluster.

## Data

The `data/` directory is not included in the repository and must be created manually. Only `dev` includes a ground-truth key and supports evaluation. `test` and `test-2026` together form the blind test set and are used for inference only.

```
data/
├── dev/                        # Development set — inference + evaluation
│   ├── archehr-qa.xml
│   └── archehr-qa_key.json
├── test/                       # Blind test set — inference only (no key)
│   └── archehr-qa.xml
└── test-2026/                  # Blind test set — inference only (no key)
    └── archehr-qa.xml
```

The data files can be obtained from the official ArchEHR-QA-2026 shared task.

## Inference

Supports local (vLLM) and cloud (OpenRouter) inference, pairwise inference, and ensemble search. Please refer to the [Inference README](./inference/README.md) for detailed instructions.

## Evaluation

Uses the official ArchEHR-QA-2026 evaluation script for answer-evidence alignment scoring. Please refer to the [Evaluation README](./evaluation/README.md) for detailed instructions.

## Other directories

| Directory    | Description                              |
| ------------ | ---------------------------------------- |
| `logs/`      | Directory to store SLURM output files.   |
| `outputs/`   | Directory to store model predictions.    |
| `results/`   | Directory to store evaluation results.   |
| `data/`      | Directory to store task data files.      |
