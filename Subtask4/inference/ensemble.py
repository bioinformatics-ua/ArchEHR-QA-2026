"""
ensemble.py — Multi-prompt ensemble for ArchEHR-QA Subtask 4

Merges predictions from multiple output JSON files using union, intersection,
or majority vote strategies.

Usage:
    python ensemble.py \
        --inputs outputs/dev/model_prompt_4.json outputs/dev/model_prompt_5.json \
        --strategy union \
        --output outputs/dev/ensemble_union.json

    python ensemble.py \
        --inputs outputs/dev/p4.json outputs/dev/p5.json outputs/dev/p11.json \
        --strategy majority \
        --majority-threshold 2 \
        --output outputs/dev/ensemble_majority.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple inference runs")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Paths to prediction JSON files to ensemble"
    )
    parser.add_argument(
        "--strategy", choices=["union", "intersection", "majority"], default="union",
        help="Merge strategy: union (any), intersection (all), majority (>=k)"
    )
    parser.add_argument(
        "--majority-threshold", type=int, default=None,
        help="For majority strategy: minimum number of votes required (default: ceil(n/2))"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save merged output JSON"
    )
    return parser.parse_args()


def load_predictions(path: str) -> dict:
    """Load a prediction file and index by case_id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["case_id"]: item["prediction"] for item in data}


def merge_predictions(
    all_preds: list[dict],
    strategy: str,
    threshold: int,
    case_id: str,
) -> list[dict]:
    """
    Merge predictions for a single case across multiple runs.
    
    all_preds: list of prediction lists (one per input file)
    Returns: merged prediction list
    """
    # Collect all answer_ids across all runs
    all_answer_ids = set()
    for preds in all_preds:
        for p in preds:
            all_answer_ids.add(str(p["answer_id"]))

    merged = []
    for answer_id in sorted(all_answer_ids, key=lambda x: int(x)):
        # Get evidence_ids per run for this answer_id
        evidence_per_run = []
        for preds in all_preds:
            run_evidence = set()
            for p in preds:
                if str(p["answer_id"]) == answer_id:
                    run_evidence = set(str(e) for e in p.get("evidence_id", []))
                    break
            evidence_per_run.append(run_evidence)

        # Collect all candidate evidence IDs
        all_evidence = set()
        for ev in evidence_per_run:
            all_evidence.update(ev)

        if strategy == "union":
            final_evidence = all_evidence

        elif strategy == "intersection":
            if evidence_per_run:
                final_evidence = evidence_per_run[0].copy()
                for ev in evidence_per_run[1:]:
                    final_evidence &= ev
            else:
                final_evidence = set()

        elif strategy == "majority":
            # Count votes per evidence_id
            vote_counts = Counter()
            for ev in evidence_per_run:
                for e in ev:
                    vote_counts[e] += 1
            final_evidence = {e for e, count in vote_counts.items() if count >= threshold}

        merged.append({
            "answer_id": answer_id,
            "evidence_id": sorted(final_evidence, key=lambda x: int(x))
        })

    return merged


def main():
    args = parse_args()

    n = len(args.inputs)
    if n < 2:
        raise ValueError("Need at least 2 input files to ensemble")

    # Determine majority threshold
    if args.strategy == "majority":
        threshold = args.majority_threshold if args.majority_threshold else (n // 2 + 1)
        print(f"Majority strategy: threshold = {threshold}/{n} votes")
    else:
        threshold = None

    print(f"Loading {n} prediction files...")
    all_loaded = []
    for path in args.inputs:
        preds = load_predictions(path)
        all_loaded.append(preds)
        print(f"  {path}: {len(preds)} cases")

    # Get union of all case_ids
    all_case_ids = set()
    for preds in all_loaded:
        all_case_ids.update(preds.keys())
    print(f"Total cases: {len(all_case_ids)}")

    # Merge predictions per case
    results = []
    for case_id in sorted(all_case_ids, key=lambda x: int(x)):
        case_preds = []
        for preds in all_loaded:
            if case_id in preds:
                case_preds.append(preds[case_id])
            else:
                print(f"  Warning: case {case_id} missing from one input file — skipping that run")
                case_preds.append([])

        merged = merge_predictions(case_preds, args.strategy, threshold, case_id)
        results.append({"case_id": case_id, "prediction": merged})

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved ensemble output ({args.strategy}) to {output_path}")

    # Print citation stats for reference
    total_citations = sum(
        len(p["evidence_id"])
        for r in results
        for p in r["prediction"]
    )
    print(f"Total citations in ensemble output: {total_citations}")


if __name__ == "__main__":
    main()