# Evaluation

This directory contains the scoring scripts for Subtask 2 of the ArchEHR-QA 2026 Shared Task, running Python 3.10+.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to set up the environment and install the required dependencies:

```bash
uv sync
```

## Running

### Single evaluation

```bash
uv run python scoring_subtask_2.py \
    --submission_path ../outputs/predictions.json \
    --key_path ../data/dev/archehr-qa.xml \
    --out_file_path ../results/scores.json
```

| Argument            | Description                          | Example Path                                  |
| ------------------- | ------------------------------------ | --------------------------------------------- |
| `--submission_path` | Path to your output predictions file | `../outputs/predictions.json`                 |
| `--key_path`        | Path to the ground truth XML file    | `../data/dev/archehr-qa.xml` (for dev set)   |
| `--out_file_path`   | Path to save the evaluation scores   | `../results/scores.json`                      |

### Batch evaluation with SLURM

```bash
sbatch evaluation.sh
```

## Results Analysis

After evaluation, analyze the results:

```bash
uv run python results_analysis.py \
    --results_file ../results/scores.json \
    --output_file ../results/analysis.json
```
