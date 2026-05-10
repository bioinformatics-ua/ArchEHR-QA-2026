"""
Run evaluations for missing result files across the 9 target models.

Scans outputs/dev for model output files and skips any that already have
a corresponding result in results/dev. Before scoring, invalid answer_ids
and evidence_ids (not present in the key) are silently filtered out so that
models which hallucinate sentence IDs still get a meaningful score instead
of crashing.

Run from: Subtask4/evaluation/
"""

from pathlib import Path
import json
from scoring_subtask_4 import load_submission, load_key, compute_alignment_scores, get_leaderboard

# ---------------------------------------------------------------------------
# Target model filename prefixes (model name with / and . replaced by -)
# ---------------------------------------------------------------------------
TARGET_PREFIXES = [
    # "google-medgemma-4b-it",
    # "google-gemma-3-27b-it",
    # "google-medgemma-1-5-4b-it",
    # "khazarai-Bio-8B-it",
    # "Qwen-Qwen3-8B",
    # "Qwen-Qwen3-32b",
    # "BioMistral-BioMistral-7B",
    # "Echelon-AI-Med-Qwen2-7B",
    # "meta-llama-Llama-3-1-8B-Instruct",
    
    'google-gemini-2-5-flash',
    'openai-gpt-4-1',
    'x-ai-grok-4-fast',
    'openai-gpt-5',
    'google-gemini-2-0-flash-001',
    'anthropic-claude-opus-4-5',
    'anthropic-claude-opus-4-6',
    'anthropic-claude-sonnet-4-5',
    'google-gemini-3-flash-preview',
    'anthropic-claude-sonnet-4',
    'x-ai-grok-4-1-fast',
    'nvidia-nemotron-3-nano-30b-a3b:free',
    'deepseek-deepseek-v3-2',
    'qwen-qwen3-max-thinking',
]

DATASET = "dev"

# Paths relative to the evaluation/ directory
EVAL_DIR = Path(__file__).parent
SUBTASK_DIR = EVAL_DIR.parent
OUTPUTS_DIR = SUBTASK_DIR / f"outputs/{DATASET}"
RESULTS_DIR = SUBTASK_DIR / f"results/{DATASET}"
KEY_PATH = SUBTASK_DIR / f"data/{DATASET}/archehr-qa_key.json"


def sanitize_submission(submission, key_map):
    """
    Remove any alignment pairs whose answer_id or evidence_id is not present
    in the reference key for that case.  Logs counts of dropped predictions.

    Returns a new submission list (original is not mutated).
    """
    sanitized = []
    total_dropped = 0
    for case in submission:
        case_id = case["case_id"]
        valid_answer_ids = key_map[case_id]["valid_answer_ids"]
        valid_evidence_ids = key_map[case_id]["valid_evidence_ids"]

        clean_prediction = []
        for alignment in case["prediction"]:
            answer_id = alignment["answer_id"]
            if answer_id not in valid_answer_ids:
                total_dropped += 1 + len(alignment["evidence_id"])
                continue  # drop the whole alignment entry

            valid_evids = [
                eid for eid in alignment["evidence_id"]
                if eid in valid_evidence_ids
            ]
            dropped = len(alignment["evidence_id"]) - len(valid_evids)
            total_dropped += dropped

            # Keep the alignment even if evidence list is now empty —
            # an empty-evidence alignment contributes 0 TP but does count
            # against precision, which is the correct penalty.
            clean_prediction.append({
                "answer_id": answer_id,
                "evidence_id": valid_evids,
            })

        sanitized.append({"case_id": case_id, "prediction": clean_prediction})

    if total_dropped:
        print(f"  [sanitize] Dropped {total_dropped} invalid id reference(s).")
    return sanitized


def score_output_file(output_file, key_map, out_file):
    """
    Load a submission file, sanitize invalid IDs, score it, and write results.
    """
    submission = load_submission(str(output_file))
    submission = sanitize_submission(submission, key_map)
    scores = compute_alignment_scores(submission, key_map)
    leaderboard = get_leaderboard(scores)

    for metric, score in leaderboard.items():
        print(f"    {metric}: {score:.2f}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(leaderboard, f, indent=2)


def main():
    outputs_dir = OUTPUTS_DIR.resolve()
    results_dir = RESULTS_DIR.resolve()
    key_path = KEY_PATH.resolve()

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    if not key_path.exists():
        raise FileNotFoundError(f"Key file not found: {key_path}")

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load key once — reused for every submission
    print("Loading reference key...")
    key_map = load_key(str(key_path))
    print(f"  {len(key_map)} cases loaded.\n")

    # Collect all output files belonging to the target models
    candidate_outputs = []
    for output_file in sorted(outputs_dir.glob("*.json")):
        stem = output_file.stem
        for prefix in TARGET_PREFIXES:
            if stem.startswith(prefix):
                candidate_outputs.append(output_file)
                break

    print(f"Found {len(candidate_outputs)} output file(s) for the target models.")

    # Find which ones are missing a result
    missing = []
    already_done = []
    for output_file in candidate_outputs:
        result_file = results_dir / output_file.name
        if result_file.exists():
            already_done.append(output_file.name)
        else:
            missing.append(output_file)

    print(f"  Already evaluated : {len(already_done)}")
    print(f"  Missing evaluations: {len(missing)}")

    if not missing:
        print("\nAll evaluations are already complete. Nothing to do.")
        return

    print("\nMissing result files:")
    for f in missing:
        print(f"  {f.name}")

    print()
    errors = []
    for i, output_file in enumerate(missing, 1):
        model_name = output_file.stem
        out_file = results_dir / output_file.name

        print(f"[{i}/{len(missing)}] Scoring: {output_file.name}")
        try:
            score_output_file(output_file, key_map, out_file)
            print(f"  -> Saved to: {out_file}\n")
        except Exception as exc:
            print(f"  ERROR scoring {model_name}: {exc}\n")
            errors.append((model_name, str(exc)))

    print("=" * 40)
    print(f"Done. {len(missing) - len(errors)}/{len(missing)} succeeded.")
    if errors:
        print(f"\nFailed ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
