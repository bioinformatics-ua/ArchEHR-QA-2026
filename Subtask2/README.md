# Subtask 2

This subdirectory contains the scripts and instructions for Subtask 2 of the ArchEHR-QA-2026 Shared Task.

It uses `uv` to manage the Python environment and dependencies. You can install it from [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Subtask 2 uses a unified Python 3.10-3.14 environment for both inference and evaluation tasks, allowing for consistent tooling and dependency management.

For both inference and evaluation, SLURM scripts are provided to facilitate running the tasks on an HPC cluster.

## Inference

It uses Python 3.10+ and supports both local inference and ensemble methods. Please refer to the [Inference README](./inference/README.md) for detailed instructions.

## Evaluation

It uses Python 3.10+ and the official ArchEHR-QA-2026 evaluation script. Please refer to the [Evaluation README](./evaluation/README.md) for detailed instructions.

## Other directories

| Directory           | Description                                    |
| ------------------- | ---------------------------------------------- |
| `logs/`             | Directory to store SLURM output files.         |
| `outputs/`          | Directory to store model predictions.          |
| `results/`          | Directory to store evaluation results.         |
| `results-analysis/` | Directory for analysis scripts and notebooks.  |
| `data/`             | Directory to store task data files.            |
