# Evaluation

This directory contains the scoring scripts for the ArchEHR-QA 2026 Shared Task, running Python 3.8.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to set up the environment and install the required dependencies:

```bash
uv sync
```

## QuickUMLS setup

Generate the QuickUMLS data files following the directions provided at <https://github.com/Georgetown-IR-Lab/QuickUMLS> (requires a UMLS license).

## Running

```bash
uv run python scoring_subtask_1.py \
    --submission_path submission.json \
    --key_path archehr-qa.xml \
    --quickumls_path quickumls/ \
    --out_file_path scores.json
```

| Argument            | Description                          | Example Path                                  |
| ------------------- | ------------------------------------ | --------------------------------------------- |
| `--submission_path` | Path to your output predictions file | `../outputs/predictions.json`                 |
| `--key_path`        | Path to the ground truth XML file    | `../../data/dev/archehr-qa.xml` (for dev set) |
| `--quickumls_path`  | Path to the QuickUMLS data directory | `../quickumls/`                               |
| `--out_file_path`   | Path to save the evaluation scores   | `../scores/scores.json`                       |
