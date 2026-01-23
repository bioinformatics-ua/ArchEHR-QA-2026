# Subtask 1

This subdirectory contains the scripts and instructions for Subtask 1 of the ArchEHR-QA-2026 Shared Task.

It uses `uv` to manage the Python environment and dependencies. You can install it from [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Because ArchEHR-QA-2026's evaluation infrastructure is quite old, we are locked on an older version of Python (3.8) for such task. Because of this, inference and evaluation virtual environments are separated, ensuring that each can run in its compatible environment.

For both inference and evaluation, SLURM scripts are provided to facilitate running the tasks on an HPC cluster.

## Pre-requisites

Besides `uv`, ensure you have the AlignScore model checkpoint downloaded on the `models/` directory:

```bash
curl -L -o ../models/AlignScore-base.ckpt https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
```

## Inference

It uses Python 3.13 and a vllm-based inference script. Please refer to the [Inference README](./inference/README.md) for detailed instructions.

## Evaluation

It uses Python 3.8 and the official ArchEHR-QA-2026 evaluation script. Please refer to the [Evaluation README](./evaluation/README.md) for detailed instructions.

## Other directories

| Directory    | Description                              |
| ------------ | ---------------------------------------- |
| `logs/`      | Directory to store SLURM output files.   |
| `outputs/`   | Directory to store model predictions.    |
| `quickumls/` | Directory to store QuickUMLS data files. |
| `results/`   | Directory to store evaluation results.   |
