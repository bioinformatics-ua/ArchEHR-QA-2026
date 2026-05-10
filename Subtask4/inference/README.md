# Inference

This directory contains the inference scripts for Subtask 4 of the ArchEHR-QA 2026 Shared Task, running Python 3.10+.

Given a clinician's answer and a clinical note, the task is to map each answer sentence to the note sentence(s) that support it.

## Dependency management

Ensure `uv` is installed on your system. You can find installation instructions at [uv's official site](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to set up the environment and install the required dependencies:

```bash
uv sync
```

Add your API credentials to a `.env` file in this directory:

```bash
OPENROUTER_API_KEY=...   # for cloud inference
HF_TOKEN=...             # for local inference with gated models
```

## Data

The inference scripts expect data files under `../data/<DATASET>/`. Only `dev` includes a ground-truth key and triggers automatic evaluation after inference. `test` and `test-2026` together form the blind test set and are used for inference only:

```
../data/
├── dev/                        # inference + auto-evaluation
│   ├── archehr-qa.xml
│   └── archehr-qa_key.json
├── test/                       # inference only (no key)
│   └── archehr-qa.xml
└── test-2026/                  # inference only (no key)
    └── archehr-qa.xml
```

## Running

### Single model inference with SLURM

Configure the variables at the top of `batch.sh`, then submit:

```bash
sbatch batch.sh
```

| Variable           | Options                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `INFERENCE_MODE`   | `local` (vLLM) / `cloud` (OpenRouter)                             |
| `INFERENCE_SCRIPT` | `standard` / `twostep` / `prefixed` / `ensemble`                  |
| `MODEL`            | see commented list in `batch.sh`                                   |
| `PROMPT_INDEX`     | integer index into `prompt.json`                                   |
| `DATASET`          | `dev` / `test-2026`                                                |

On `dev`, evaluation runs automatically after inference.

## Inference Variants

| Script                   | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `inference.py`           | Default — processes all answer sentences at once                       |
| `inference_twostep.py`   | Adds clinical knowledge filtering + optional second-pass verification  |
| `inference_prefixed.py`  | Experimental prefix-based prompting (`[N1]`/`[A1]` sentence IDs)      |

## Pairwise Inference

The pairwise variant has its own batch script and prompt file:

```bash
sbatch batch_pairwise.sh
```

Configure model and prompt index at the top of `batch_pairwise.sh`. Uses `prompt_subtask4.json` instead of `prompt.json`.

## Ensemble Search

To find the best combination of model outputs:

```bash
sbatch find_best_ensemble.sh run1
```

To manually ensemble specific outputs, set `INFERENCE_SCRIPT="ensemble"` in `batch.sh`.
