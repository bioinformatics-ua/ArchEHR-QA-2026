# Inference

This directory contains the inference scripts for Subtask 2 of the ArchEHR-QA 2026 Shared Task, running Python 3.10+.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to set up the environment and install the required dependencies:

```bash
uv sync
```

## Running

### Single model inference

```bash
uv run python inference.py \
    --xml_file path/to/archehr-qa.xml \
    --output_file ../outputs/predictions.json \
    --model Qwen/Qwen3-8B
```

| Argument        | Description                                         | Example Path                                  |
| --------------- | --------------------------------------------------- | --------------------------------------------- |
| `--xml_file`    | Path to the input XML file containing the questions | `../data/dev/archehr-qa.xml` (for dev set)   |
| `--output_file` | Path to save the output predictions file            | `../outputs/predictions.json`                 |
| `--model`       | Model name or path for inference                    | `Qwen/Qwen3-8B`                               |

### Batch inference with SLURM

```bash
sbatch inference.sh
```

### Ensemble inference

For running ensemble inference across multiple models:

```bash
uv run python ensemble.py \
    --xml_file path/to/archehr-qa.xml \
    --output_file ../outputs/ensemble_predictions.json \
    --models model1 model2 model3
```

### Deduplication

After inference, you can deduplicate results:

```bash
uv run python deduplicate.py \
    --input_file ../outputs/predictions.json \
    --output_file ../outputs/predictions_deduplicated.json
```
