# ArchEHR-QA-2026

### Evaluation Setup (Quick Start)

Run these commands from the `Subtask1` directory to set up your environment and download the required model checkpoint.

```bash
# 1. Create and activate environment (Python 3.10)
uv venv --python 3.10 venv-eval
source venv-eval/bin/activate

# 2. Install dependencies
uv pip install -r requirements-eval.txt

# 3. Download AlignScore Checkpoint
# Ensure the destination path matches your project structure
curl -L -o ../models/AlignScore-base.ckpt https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt

```

**Usage:**

```bash
python evaluate.py path/to/your_predictions.json
``` 