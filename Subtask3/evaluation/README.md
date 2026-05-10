# Evaluation

This directory contains the scoring scripts for Subtask 3 of the ArchEHR-QA 2026 Shared Task, running via a Singularity container.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

The evaluation runs inside a Singularity container. Build or pull it with:

```bash
srun singularity pull builder.sif docker://python:3.10-bookworm
```

Then install the Python dependencies inside the container:

```bash
singularity exec builder.sif uv sync
```

## QuickUMLS setup

The scorer requires a QuickUMLS data directory. Generate the QuickUMLS data files following the directions at <https://github.com/Georgetown-IR-Lab/QuickUMLS> (requires a UMLS license), then place the result at `./quickumls/final/`.

## Running

### Batch evaluation with SLURM

Configure `DATASET` at the top of `evaluation.sh`, then submit:

```bash
sbatch evaluation.sh
```

This iterates over all prediction files in `../outputs/<DATASET>/` and scores each one, skipping files that already have a result in `../results/<DATASET>/`.

### Single evaluation

```bash
singularity exec --nv builder.sif uv run python scoring_subtask_3.py \
    --submission_path ../outputs/dev/predictions.json \
    --key_path ../data/dev/archehr-qa_key.json \
    --data_path ../data/dev/archehr-qa.xml \
    --quickumls_path ./quickumls/final \
    --out_file_path ../results/dev/scores.json \
    --case_ids_to_score 1-3
```

| Argument              | Description                                  | Example Path                              |
| --------------------- | -------------------------------------------- | ----------------------------------------- |
| `--submission_path`   | Path to your output predictions file         | `../outputs/dev/predictions.json`         |
| `--key_path`          | Path to the ground truth key JSON file       | `../data/dev/archehr-qa_key.json`         |
| `--data_path`         | Path to the input XML file                   | `../data/dev/archehr-qa.xml`              |
| `--quickumls_path`    | Path to the QuickUMLS data directory         | `./quickumls/final`                       |
| `--out_file_path`     | Path to save the evaluation scores           | `../results/dev/scores.json`              |
| `--case_ids_to_score` | Range of case IDs to evaluate                | `1-3`                                     |
