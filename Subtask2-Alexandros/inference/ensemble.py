"""
Self-consistency ensembling for Subtask 2: Evidence Identification

Runs the same prompt N times at a given temperature and takes the union of all
predicted sentence IDs per case. Union ensembling maximises recall — a sentence
only needs to be picked by *one* run to survive — which directly improves strict
F1 when recall is the bottleneck.

Usage (via ensemble.sh):
    uv run python ensemble.py \
        --xml-file ../../data/dev/archehr-qa.xml \
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

from dataloader import ArchEHRSubtask2DataLoader


def parse_prediction(output: str, case_id: str, valid_ids: set, provider) -> list:
    """Extract a list of sentence ID strings from a raw LLM response."""
    parsed = provider.parse_response(output)
    if parsed:
        prediction = parsed.get("prediction", [])
    else:
        json_match = re.search(r"(\[.*?\]|\{.*?\})", output, re.DOTALL)
        if json_match:
            try:
                data = orjson.loads(json_match.group())
                if isinstance(data, list):
                    data = next(
                        (obj for obj in data
                         if isinstance(obj, dict)
                         and str(obj.get("case_id")) == str(case_id)),
                        data[0] if data else {}
                    )
                prediction = data.get("prediction", [])
            except Exception:
                prediction = []
        else:
            print(f"  Warning: No JSON found for case {case_id}")
            prediction = []

    if not isinstance(prediction, list):
        prediction = [prediction]

    # Validate — keep only real sentence IDs
    clean = []
    for p_id in prediction:
        digits = re.findall(r"\d+", str(p_id))
        if digits and digits[0] in valid_ids:
            clean.append(digits[0])
    return clean


def main():
    parser = argparse.ArgumentParser("Subtask 2: Self-consistency ensembling")
    parser.add_argument("--xml-file",       type=Path, required=True)
    parser.add_argument("--prompt-file",    type=Path, required=True)
    parser.add_argument("--prompt-index",   type=int,  required=True)
    parser.add_argument("--output-file",    type=Path, required=True)
    parser.add_argument("--inference-mode", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--model",          type=str,  required=True)
    parser.add_argument("--n-runs",         type=int,  default=5,
                        help="Number of independent inference passes")
    parser.add_argument("--min-votes",      type=int,  default=1,
                        help="Min runs a sentence must appear in to be kept (1=union, n-runs=intersection)")
    # Sampling
    parser.add_argument("--temperature",         type=float, default=0.7)
    parser.add_argument("--top-p",               type=float, default=0.95)
    parser.add_argument("--max-tokens",          type=int,   default=512)
    parser.add_argument("--repetition-penalty",  type=float, default=1.0)
    # Local-only
    parser.add_argument("--tensor-parallel-size",    type=int,   default=1)
    parser.add_argument("--gpu-memory-utilization",  type=float, default=0.95)
    parser.add_argument("--max-model-len",           type=int,   default=4096)
    args = parser.parse_args()

    # --- Provider ---
    if args.inference_mode == "cloud":
        from providers.cloud import CloudProvider
        provider = CloudProvider(
            args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    else:
        from providers.local import LocalProvider
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

    # --- Data ---
    print(f"Loading XML cases from {args.xml_file}...")
    cases = ArchEHRSubtask2DataLoader(args.xml_file).load()
    print(f"Loaded {len(cases)} cases.")

    # --- Prompt ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)
    prompt_template = prompt_dict[str(args.prompt_index)]

    # ============================================================
    # N-RUN INFERENCE → UNION
    # ============================================================
    # accumulated[case_id] = Counter of how many runs predicted each sentence ID
    accumulated = defaultdict(Counter)
    per_run_results = []

    for run in range(1, args.n_runs + 1):
        print(f"\n{'='*50}")
        print(f"Run {run}/{args.n_runs}  (temperature={args.temperature})")
        print(f"{'='*50}")

        run_preds = []
        for case in cases:
            sentences_text = "\n".join(
                f"ID {s['sentence_id']}: {s['text']}"
                for s in case["sentences"]
            )
            payload = {
                "clinician_question":  case["clinician_question"],
                "patient_question":    case.get("patient_question", ""),
                "patient_narrative":   case.get("patient_narrative", ""),
                "clinical_specialty":  case.get("clinical_specialty", ""),
                "numbered_sentences":  sentences_text,
                "sentences":           sentences_text,
                "case_id":             case["case_id"],
            }

            prompt = provider.build_prompt(prompt_template, payload)
            output = provider.generate(prompt)

            valid_ids = {str(s["sentence_id"]) for s in case["sentences"]}
            prediction = parse_prediction(output, case["case_id"], valid_ids, provider)

            run_preds.append({"case_id": case["case_id"], "prediction": prediction})
            for sid in prediction:
                accumulated[case["case_id"]][sid] += 1
            n_seen = sum(1 for v in accumulated[case["case_id"]].values() if v >= args.min_votes)
            print(f"  case {case['case_id']}: {len(prediction)} sentences  (passing min_votes={args.min_votes} so far: {n_seen})")

        per_run_results.append(run_preds)

    # ============================================================
    # BUILD UNION SUBMISSION
    # ============================================================
    submission = []
    for case in cases:
        cid = case["case_id"]
        kept_ids = sorted(
            (sid for sid, count in accumulated[cid].items() if count >= args.min_votes),
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        submission.append({"case_id": str(cid), "prediction": kept_ids})
        print(f"Case {cid}: kept {len(kept_ids)} sentences (min_votes={args.min_votes}) = {kept_ids}")

    # ============================================================
    # SAVE
    # ============================================================
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    # Save full per-run breakdown alongside the output
    breakdown_path = output_path.parent / (output_path.stem + "_per_run.json")
    with open(breakdown_path, "w") as f:
        json.dump(per_run_results, f, indent=2)

    print(f"\n[DONE] Ensemble complete ({args.n_runs} runs, min_votes={args.min_votes}).")
    print(f"  Union output  : {output_path}")
    print(f"  Per-run detail: {breakdown_path}")


if __name__ == "__main__":
    main()
