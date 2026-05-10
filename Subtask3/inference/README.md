# Inference

This directory contains the inference scripts for Subtask 3 of the ArchEHR-QA 2026 Shared Task, running Python 3.10+.

Given a patient narrative, a clinician question, and a clinical note, the task is to generate a concise evidence-grounded answer (max 75 words).

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

The inference scripts expect data files under `../data/<DATASET>/`. Three splits are used — `dev` and `test` include a ground-truth key and support automatic evaluation after inference, while `test-2026` is the blind submission set (no key):

```
../data/
├── dev/                        # inference + auto-evaluation
│   ├── archehr-qa.xml
│   └── archehr-qa_key.json
├── test/                       # inference + auto-evaluation
│   ├── archehr-qa.xml
│   └── archehr-qa_key.json
└── test-2026/                  # inference only (no key)
    └── archehr-qa.xml
```

## Running

### Single model inference with SLURM

Configure the variables at the top of `inference.sh`, then submit:

```bash
sbatch inference.sh
```

| Variable         | Options                                   |
| ---------------- | ----------------------------------------- |
| `INFERENCE_MODE` | `local` (vLLM) / `cloud` (OpenRouter)    |
| `MODEL`          | see commented list in `inference.sh`      |
| `PROMPT_INDEX`   | integer index into `prompt.json`          |
| `DATASET`        | `dev` / `test` / `test-2026`             |

On `dev` and `test`, evaluation runs automatically after inference using a Singularity container.

### Batch inference with SLURM

To sweep across multiple models and prompts automatically:

```bash
sbatch cloud_batch.sh        # cloud models × all prompts
sbatch open_batch.sh 1       # open-source model 1 × all prompts
sbatch open_batch.sh 2       # open-source model 2 × all prompts
```

Each `open_batch.sh` job runs one model — submit multiple in parallel.

## Ensemble

Configure candidate files and judge model at the top of `ensemble.sh`, then submit:

```bash
sbatch ensemble.sh
```

| Variable | Options                                                          |
| -------- | ---------------------------------------------------------------- |
| `MODE`   | `select` (pick best candidate) / `merge` (synthesise new answer) |

The judge model selects or merges answers from the candidate output files.
