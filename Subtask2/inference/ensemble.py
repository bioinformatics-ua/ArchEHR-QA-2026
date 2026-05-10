"""
Self-consistency ensembling for Subtask 2: Evidence Identification

Runs the same prompt N times at a given temperature and aggregates
predicted sentence IDs across runs.

Usage:
    uv run python -m Subtask2.inference.ensemble \
        --xml-file ../data/dev/archehr-qa.xml \
        --prompt-file prompt.json \
        --prompt-index 5 \
        --output-file ../outputs/dev/ensemble.json \
        --inference-mode cloud \
        --model google/gemini-2.5-flash \
        --n-runs 5 \
        --temperature 0.7 \
        --max-tokens 512
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import orjson

from Subtask2.inference.dataloader import ArchEHRSubtask2DataLoader
from Subtask2.inference.providers.cloud import CloudProvider
from Subtask2.inference.providers.local import LocalProvider


def parse_prediction(
    output: str,
    case_id: str,
    valid_ids: set[str],
    provider,
) -> list[str]:
    """
    Extract and clean sentence IDs from raw model output.
    """

    parsed = provider.parse_response(output)

    if parsed:

        prediction = parsed.get("prediction", [])

    else:

        json_match = re.search(
            r"(\[.*?\]|\{.*?\})",
            output,
            re.DOTALL,
        )

        if json_match:

            try:

                data = orjson.loads(json_match.group())

                if isinstance(data, list):

                    data = next(
                        (
                            obj for obj in data
                            if isinstance(obj, dict)
                            and str(obj.get("case_id")) == str(case_id)
                        ),
                        data[0] if data else {},
                    )

                prediction = data.get("prediction", [])

            except Exception:

                prediction = []

        else:

            print(f"  Warning: No JSON found for case {case_id}")

            prediction = []

    # Ensure list
    if not isinstance(prediction, list):
        prediction = [prediction]

    # Keep only valid sentence IDs
    clean_prediction = []

    for p_id in prediction:

        digits = re.findall(r"\d+", str(p_id))

        if digits and digits[0] in valid_ids:
            clean_prediction.append(digits[0])

    return clean_prediction


def main():

    parser = argparse.ArgumentParser(
        "Subtask 2: Self-consistency ensembling"
    )

    parser.add_argument("--xml-file", type=Path, required=True)

    parser.add_argument("--prompt-file", type=Path, required=True)

    parser.add_argument("--prompt-index", type=int, required=True)

    parser.add_argument("--output-file", type=Path, required=True)

    parser.add_argument(
        "--inference-mode",
        choices=["local", "cloud"],
        default="cloud",
    )

    parser.add_argument("--model", type=str, required=True)

    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of independent inference runs",
    )

    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help=(
            "Minimum number of runs a sentence must appear in "
            "to survive. "
            "(1=union, n-runs=intersection)"
        ),
    )

    # ============================================================
    # SAMPLING
    # ============================================================

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
    )

    # ============================================================
    # LOCAL MODEL SETTINGS
    # ============================================================

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )

    args = parser.parse_args()

    # ============================================================
    # PROVIDER
    # ============================================================

    if args.inference_mode == "cloud":

        provider = CloudProvider(
            args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

    else:

        provider = LocalProvider(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
        )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print(f"Loading XML cases from {args.xml_file}...")

    cases = ArchEHRSubtask2DataLoader(args.xml_file).load()

    print(f"Loaded {len(cases)} cases.")

    # ============================================================
    # LOAD PROMPT
    # ============================================================

    with args.prompt_file.open("r") as f:
        prompt_dict = json.load(f)

    prompt_template = prompt_dict[str(args.prompt_index)]

    # ============================================================
    # ENSEMBLE INFERENCE
    # ============================================================

    accumulated = defaultdict(Counter)

    per_run_results = []

    for run in range(1, args.n_runs + 1):

        print(f"\n{'=' * 50}")
        print(
            f"Run {run}/{args.n_runs} "
            f"(temperature={args.temperature})"
        )
        print(f"{'=' * 50}")

        run_predictions = []

        for case in cases:

            sentences_text = "\n".join(
                f"ID {s['sentence_id']}: {s['text']}"
                for s in case["sentences"]
            )

            payload = {
                "clinician_question": case["clinician_question"],
                "patient_question": case.get(
                    "patient_question",
                    "",
                ),
                "patient_narrative": case.get(
                    "patient_narrative",
                    "",
                ),
                "clinical_specialty": case.get(
                    "clinical_specialty",
                    "",
                ),
                "numbered_sentences": sentences_text,
                "sentences": sentences_text,
                "case_id": case["case_id"],
            }

            prompt = provider.build_prompt(
                prompt_template,
                payload,
            )

            output = provider.generate(prompt)

            valid_ids = {
                str(s["sentence_id"])
                for s in case["sentences"]
            }

            prediction = parse_prediction(
                output=output,
                case_id=case["case_id"],
                valid_ids=valid_ids,
                provider=provider,
            )

            run_predictions.append({
                "case_id": case["case_id"],
                "prediction": prediction,
            })

            for sentence_id in prediction:
                accumulated[case["case_id"]][sentence_id] += 1

            kept_so_far = sum(
                1
                for count in accumulated[
                    case["case_id"]
                ].values()
                if count >= args.min_votes
            )

            print(
                f"  case {case['case_id']}: "
                f"{len(prediction)} sentences "
                f"(passing min_votes={args.min_votes}: "
                f"{kept_so_far})"
            )

        per_run_results.append(run_predictions)

    # ============================================================
    # BUILD FINAL SUBMISSION
    # ============================================================

    submission = []

    for case in cases:

        case_id = case["case_id"]

        kept_ids = sorted(
            (
                sentence_id
                for sentence_id, count
                in accumulated[case_id].items()
                if count >= args.min_votes
            ),
            key=lambda x: int(x) if x.isdigit() else 0,
        )

        submission.append({
            "case_id": str(case_id),
            "prediction": kept_ids,
        })

        print(
            f"Case {case_id}: "
            f"kept {len(kept_ids)} sentences "
            f"(min_votes={args.min_votes}) "
            f"= {kept_ids}"
        )

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    output_path = Path(args.output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(submission, f, indent=2)

    # Save per-run predictions
    breakdown_path = output_path.parent / (
        output_path.stem + "_per_run.json"
    )

    with breakdown_path.open("w") as f:
        json.dump(per_run_results, f, indent=2)

    print(
        f"\n[DONE] Ensemble complete "
        f"({args.n_runs} runs, "
        f"min_votes={args.min_votes})."
    )

    print(f"  Submission : {output_path}")

    print(f"  Per-run    : {breakdown_path}")


if __name__ == "__main__":
    main()