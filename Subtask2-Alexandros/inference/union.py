"""
Union-of-Experts for Subtask 2: Evidence Identification

Runs each prompt in PROMPT_INDICES once and takes the union of predictions
per case. Different prompts act as "experts" specialised in different
reasoning patterns (e.g. keyword extraction vs. causal chain detection).

Usage (via union.sh):
    uv run python union.py \
        --xml-file ../../data/dev/archehr-qa.xml \
        --prompt-file prompt.json \
        --prompt-indices 9 4 \
        --output-file ../outputs/dev/union_p9_p4.json \
        --inference-mode cloud \
        --model google/gemini-2.5-flash \
        --temperature 0.0 \
        --max-tokens 512
"""

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import orjson

from dataloader import ArchEHRSubtask2DataLoader


def parse_prediction(output: str, case_id: str, valid_ids: set, provider) -> list:
    """Extract sentence ID strings from a raw LLM response."""
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

    clean = []
    for p_id in prediction:
        digits = re.findall(r"\d+", str(p_id))
        if digits and digits[0] in valid_ids:
            clean.append(digits[0])
    return clean



def main():
    parser = argparse.ArgumentParser("Subtask 2: Union-of-Experts")
    parser.add_argument("--xml-file",        type=Path, required=True)
    parser.add_argument("--prompt-file",     type=Path, required=True)
    parser.add_argument("--prompt-indices",  type=int, nargs="+", required=True,
                        help="Prompt indices to use (e.g. 9 4 or just 3 for filter mode)")
    parser.add_argument("--output-file",     type=Path, required=True)
    parser.add_argument("--inference-mode",  choices=["local", "cloud"], default="cloud")
    parser.add_argument("--model",           type=str, required=True)
    # Sampling
    parser.add_argument("--temperature",        type=float, default=0.0)
    parser.add_argument("--top-p",              type=float, default=0.95)
    parser.add_argument("--max-tokens",         type=int,   default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    # Local-only
    parser.add_argument("--tensor-parallel-size",   type=int,   default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len",          type=int,   default=4096)
    # Filtering mode
    parser.add_argument("--filter-predictions", type=Path, required=False,
                        help="If set, filter these predictions using the given prompt index (must be one index)")
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

    # --- Prompts ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)

    # --- Data ---
    print(f"Loading XML cases from {args.xml_file}...")
    cases = ArchEHRSubtask2DataLoader(args.xml_file).load()
    print(f"Loaded {len(cases)} cases.")

    # ...existing code...

    # ============================================================
    # NORMAL UNION MODE (no filter)
    # ============================================================
    if len(args.prompt_indices) < 2:
        raise ValueError("--prompt-indices requires at least 2 values (unless using --filter-predictions)")

    for idx in args.prompt_indices:
        if str(idx) not in prompt_dict:
            raise KeyError(f"Prompt index {idx} not found in {args.prompt_file}")

    accumulated = defaultdict(set)
    per_prompt_results = {}   # {prompt_idx: [{case_id, prediction}]}

    for prompt_idx in args.prompt_indices:
        prompt_template = prompt_dict[str(prompt_idx)]
        print(f"\n{'='*50}")
        print(f"Expert: prompt {prompt_idx}")
        print(f"{'='*50}")

        prompt_preds = []
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

            prompt_preds.append({"case_id": case["case_id"], "prediction": prediction})
            accumulated[case["case_id"]].update(prediction)
            print(f"  case {case['case_id']}: {len(prediction)} sentences  "
                  f"(union so far: {len(accumulated[case['case_id']])})")

        per_prompt_results[prompt_idx] = prompt_preds

    submission = []
    for case in cases:
        cid = case["case_id"]
        union_ids = sorted(
            accumulated[cid],
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        submission.append({"case_id": str(cid), "prediction": union_ids})
        print(f"Case {cid}: union = {union_ids}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    breakdown_path = output_path.parent / (output_path.stem + "_per_prompt.json")
    with open(breakdown_path, "w") as f:
        json.dump(per_prompt_results, f, indent=2)

    indices_str = "+".join(str(i) for i in args.prompt_indices)
    print(f"\n[DONE] Union-of-Experts complete (prompts {indices_str}).")
    print(f"  Union output    : {output_path}")
    print(f"  Per-prompt detail: {breakdown_path}")


if __name__ == "__main__":
    main()
