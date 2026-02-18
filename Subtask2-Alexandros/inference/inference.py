import re
import json
import argparse
from pathlib import Path

import orjson

from dataloader import ArchEHRSubtask2DataLoader


def main():
    parser = argparse.ArgumentParser("Subtask 2: Evidence Identification inference")
    parser.add_argument("--xml-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--inference-mode", choices=["local", "cloud"], default="local"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Context window in tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.90,
        help="Nucleus sampling cutoff",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per case",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (>1.0 discourages repeats)",
    )
    args = parser.parse_args()

    # --- Provider (lazy import to avoid loading vLLM when using cloud) ---
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

    # --- Load data ---
    print(f"Loading XML cases from {args.xml_file}...")
    cases = ArchEHRSubtask2DataLoader(args.xml_file).load()
    print(f"Loaded {len(cases)} cases.")

    # --- Load prompt template ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)
    prompt_template = prompt_dict[str(args.prompt_index)]

    # ============================================================
    # INFERENCE
    # ============================================================
    raw_results = []

    for case in cases:
        sentences_text = "\n".join(
            f"ID {s['sentence_id']}: {s['text']}"
            for s in case["sentences"]
        )

        payload = {
            "clinician_question": case["clinician_question"],
            "patient_question": case.get("patient_question", ""),
            "patient_narrative": case.get("patient_narrative", ""),
            "clinical_specialty": case.get("clinical_specialty", ""),
            "numbered_sentences": sentences_text,
            "sentences": sentences_text,
            "case_id": case["case_id"],
        }

        prompt = provider.build_prompt(prompt_template, payload)
        output = provider.generate(prompt)

        # Parse JSON response
        parsed = provider.parse_response(output)
        if parsed:
            prediction = parsed.get("prediction", [])
        else:
            # Fallback: try to extract any JSON array or object
            json_match = re.search(r"(\[.*?\]|\{.*?\})", output, re.DOTALL)
            if json_match:
                try:
                    data = orjson.loads(json_match.group())
                    if isinstance(data, list):
                        data = next(
                            (obj for obj in data
                             if isinstance(obj, dict)
                             and str(obj.get("case_id")) == str(case["case_id"])),
                            data[0] if data else {}
                        )
                    prediction = data.get("prediction", [])
                except Exception:
                    prediction = []
            else:
                print(f"  Warning: No JSON found for case {case['case_id']}")
                prediction = []

        if not isinstance(prediction, list):
            prediction = [prediction]

        # Validate IDs against actual sentence IDs
        valid_ids = {str(s["sentence_id"]) for s in case["sentences"]}
        clean_prediction = []
        for p_id in prediction:
            digits = re.findall(r"\d+", str(p_id))
            if digits and digits[0] in valid_ids:
                clean_prediction.append(digits[0])

        raw_results.append({
            "case_id": case["case_id"],
            "prediction": clean_prediction,
            "raw_output": output,
        })

        print(f"  Finished case {case['case_id']}")

    # ============================================================
    # GROUPING FOR SUBMISSION
    # ============================================================
    submission = []
    for r in raw_results:
        unique_ids = sorted(
            list(set(r["prediction"])),
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        submission.append({
            "case_id": str(r["case_id"]),
            "prediction": unique_ids,
        })

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"[DONE] Subtask 2 inference finished.\nOutput: {output_path}")


if __name__ == "__main__":
    main()
