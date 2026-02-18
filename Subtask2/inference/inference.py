import re
import orjson
import json
import argparse
from pathlib import Path

from dataloader import ArchEHRSubtask2DataLoader


def main():

    parser = argparse.ArgumentParser("Subtask 2 inference")
    parser.add_argument("--xml-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--inference-mode", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    args = parser.parse_args()

    # --- Provider ---
    if args.inference_mode == "cloud":
        from providers.cloud import CloudProvider
        provider = CloudProvider(args.model)
    else:
        from providers.local import LocalProvider
        provider = LocalProvider(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization
        )

    # --- Load data ---
    loader = ArchEHRSubtask2DataLoader(args.xml_file)
    cases = loader.load()

    # --- Load prompt ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)
    prompt_template = prompt_dict[str(args.prompt_index)]

    results = []

    # ============================================================
    # 🔵 SENTENCE-BASED (PROMPTS 0,1)
    # ============================================================
    if args.prompt_index in [0, 1]:

        for case in cases:
            for sent in case["sentences"]:

                payload = {
                    "clinician_question": case["clinician_question"],
                    "patient_question": case.get("patient_question", ""),
                    "sentence": sent["text"],
                }

                prompt = provider.build_prompt(prompt_template, payload)
                output = provider.generate(prompt)
                label = provider.parse_response(output)

                results.append({
                    "case_id": case["case_id"],
                    "sentence_id": str(sent["sentence_id"]),
                    "label": label,
                })

    # ============================================================
    # 🟢 CASE-BASED (PROMPTS 2–7)
    # ============================================================
    else:

        for case in cases:

            sentences_text = "\n".join(
                [f"ID {s['sentence_id']}: {s['text']}" for s in case["sentences"]]
            )

            payload = {
                "clinician_question": case["clinician_question"],
                "patient_question": case.get("patient_question", ""),
                "sentences": sentences_text,
                "case_id": case["case_id"]
            }

            prompt = provider.build_prompt(prompt_template, payload)
            output = provider.generate(prompt)

            try:
                json_match = re.search(r"(\[.*\]|\{.*\})", output, re.DOTALL)

                if not json_match:
                    print(f"Warning: No JSON found for case {case['case_id']}")
                    continue

                data = orjson.loads(json_match.group())

                if isinstance(data, list):
                    data = next(
                        (obj for obj in data
                         if isinstance(obj, dict)
                         and str(obj.get("case_id")) == str(case["case_id"])),
                        data[0] if data else {}
                    )

                prediction = data.get("prediction", [])

                if not isinstance(prediction, list):
                    prediction = [prediction]

                valid_ids = {str(s["sentence_id"]) for s in case["sentences"]}

                for p_id in prediction:

                    digits = re.findall(r"\d+", str(p_id))
                    if not digits:
                        continue

                    p_id = digits[0]

                    if p_id not in valid_ids:
                        continue

                    results.append({
                        "case_id": case["case_id"],
                        "sentence_id": p_id,
                        "label": "essential"
                    })

            except Exception as e:
                print(f"Parse error case {case['case_id']}: {e}")

    # ============================================================
    # 🔥 GROUPING FOR SUBMISSION
    # ============================================================

    grouped_results = []
    case_map = {str(c["case_id"]): [] for c in cases}

    for r in results:
        if r["label"] == "essential":
            case_map[str(r["case_id"])].append(str(r["sentence_id"]))

    for case_id, sentence_ids in case_map.items():
        unique_ids = sorted(
            list(set(sentence_ids)),
            key=lambda x: int(x) if x.isdigit() else 0
        )

        grouped_results.append({
            "case_id": str(case_id),
            "prediction": unique_ids
        })

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    output_path = Path(args.output_file)
    split = output_path.parent.name
    fname = output_path.name

    base_dir = Path(__file__).resolve().parent.parent
    output_file = base_dir / "outputs" / split / fname
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(grouped_results, f, indent=2)

    analysis_file = base_dir / "analysis" / split / fname
    analysis_file.parent.mkdir(parents=True, exist_ok=True)

    with open(analysis_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Subtask 2 inference finished.")
    print(f"Submission file: {output_file}")
    print(f"Analysis file: {analysis_file}")


if __name__ == "__main__":
    main()