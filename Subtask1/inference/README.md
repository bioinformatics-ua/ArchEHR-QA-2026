# Inference

This directory contains the inference script for the ArchEHR-QA 2026 Shared Task, running Python 3.13.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to set up the environment and install the required dependencies:

```bash
uv sync
```

## Running

```bash
uv run python inference.py \
    --xml_file path/to/archehr-qa.xml \
    --prompt_file prompt.jsonl \
    --output_file ../outputs/predictions.json \
    --model Qwen/Qwen3-8B
```

| Argument        | Description                                         | Example Path                                  |
| --------------- | --------------------------------------------------- | --------------------------------------------- |
| `--xml_file`    | Path to the input XML file containing the questions | `../../data/dev/archehr-qa.xml` (for dev set) |
| `--prompt_file` | Path to the prompt template JSONL file              | `prompt.jsonl`                                |
| `--output_file` | Path to save the output predictions file            | `../outputs/predictions.json`                 |
| `--model`       | Model name or path for inference                    | `Qwen/Qwen3-8B`                               |
