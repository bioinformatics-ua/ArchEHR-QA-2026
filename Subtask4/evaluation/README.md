# Evaluation

This directory contains the scoring scripts for Subtask 4 of the ArchEHR-QA 2026 Shared Task, running Python 3.10+.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to set up the environment and install the required dependencies:

```bash
uv sync
```

## Running

### Single evaluation with SLURM

Configure `MODEL` and `DATASET` at the top of `evaluation.sh`, then submit:

```bash
sbatch evaluation.sh
```

### Single evaluation

```bash
uv run python scoring_subtask_4.py \
    --submission_path ../outputs/dev/predictions.json \
    --key_path ../data/dev/archehr-qa_key.json \
    --out_file_path ../results/dev/scores.json
```

| Argument            | Description                                  | Example Path                              |
| ------------------- | -------------------------------------------- | ----------------------------------------- |
| `--submission_path` | Path to your output predictions file         | `../outputs/dev/predictions.json`         |
| `--key_path`        | Path to the ground truth key JSON file       | `../data/dev/archehr-qa_key.json`         |
| `--out_file_path`   | Path to save the evaluation scores           | `../results/dev/scores.json`              |

### Batch evaluation

To re-run evaluation for any missing result files:

```bash
sbatch run_missing_evaluations.sh
```

Or run the script directly:

```bash
uv run python run_missing_evaluations.py
```
