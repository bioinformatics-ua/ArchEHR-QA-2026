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
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor parallel GPUs to use")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization (0.0-1.0)")

    args = parser.parse_args()

    # --- Provider (IMPORT LAZY, SEM vllm NO CLOUD) ---
    if args.inference_mode == "cloud":
        from providers.cloud import CloudProvider
        provider = CloudProvider(args.model)
    else:
        from providers.local import LocalProvider
        provider = LocalProvider(args.model, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)

    # --- Load data ---
    loader = ArchEHRSubtask2DataLoader(args.xml_file)
    cases = loader.load()

    # --- Load prompt ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)
    prompt_template = prompt_dict[str(args.prompt_index)]

    results = []

    if args.prompt_index in [2, 3, 4, 5, 6]:
        # Case-based inference
        for case in cases:
            sentences_text = "\n".join([f"ID {s['sentence_id']}: {s['text']}" for s in case["sentences"]])
            payload = {
                "clinician_question": case["clinician_question"],
                "patient_question": case.get("patient_question", ""),
                "sentences": sentences_text,
                "case_id": case["case_id"]
            }
            prompt = provider.build_prompt(prompt_template, payload)
            output = provider.generate(prompt)
            
            # Simple heuristic to extract the prediction list from JSON
            try:
                # Look for lists or objects
                json_match = re.search(r"(\[.*\]|\{.*\})", output, re.DOTALL)
                if json_match:
                    try:
                        data = orjson.loads(json_match.group())
                        
                        # Handle if model returns a list of objects instead of one object
                        if isinstance(data, list):
                            # Filter to find the object matching CURRENT case_id if possible
                            matching_objs = [obj for obj in data if isinstance(obj, dict) and str(obj.get("case_id")) == str(case["case_id"])]
                            if matching_objs:
                                prediction = matching_objs[0].get("prediction", [])
                            else:
                                # Fallback to first object if no ID matches
                                prediction = data[0].get("prediction", []) if len(data) > 0 and isinstance(data[0], dict) else []
                        else:
                            prediction = data.get("prediction", [])
                            
                        if not isinstance(prediction, list):
                            prediction = [prediction]
                        
                        for p_id in prediction:
                            # Sanitize: extract digits only if it's like "ID 2" or "sentence 2"
                            if isinstance(p_id, (str, int)):
                                digits = re.findall(r"\d+", str(p_id))
                                if digits:
                                    p_id = digits[0]
                            
                            results.append({
                                "case_id": case["case_id"],
                                "sentence_id": str(p_id),
                                "label": "essential" 
                            })
                    except Exception as json_err:
                        print(f"JSON load error for case {case['case_id']}: {json_err}. Output snippet: {output[:100]}")
                else:
                    print(f"Warning: No JSON structure found in output for case {case['case_id']}")
            except Exception as e:
                print(f"Error parsing response for case {case['case_id']}: {e}")
    else:
        # Original sentence-based inference
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

                results.append(
                    {
                        "case_id": case["case_id"],
                        "sentence_id": sent["sentence_id"],
                        "label": label,
                    }
                )

    # --- Submission output: grouped by case_id ---
    grouped_results = []
    case_map = {str(c["case_id"]): [] for c in cases}
    for r in results:
        # Convert sentence_id to string for grouping
        if r["label"] == "essential":
            case_map.setdefault(str(r["case_id"]), []).append(str(r["sentence_id"]))
    
    for case_id, sentence_ids in case_map.items():
        # Sort sentence IDs numerically for consistent output format
        # Filter for duplicates and ensure only valid IDs are included
        unique_ids = sorted(list(set(sentence_ids)), key=lambda x: int(x) if x.isdigit() else 0)
        grouped_results.append({
            "case_id": str(case_id),
            "prediction": unique_ids
        })


    def custom_json_dump_horizontal_predictions(data, file):
        file.write('[\n')
        for i, obj in enumerate(data):
            file.write('  {\n')
            file.write(f'    "case_id": "{obj["case_id"]}",\n')
            preds = ', '.join(f'"{sid}"' for sid in obj["prediction"])
            file.write(f'    "prediction": [{preds}]\n')
            file.write('  }')
            if i < len(data) - 1:
                file.write(',\n')
            else:
                file.write('\n')
        file.write(']\n')

    # --- Simple: use only last two parts (split and filename) ---
    output_path = Path(args.output_file)
    split = output_path.parent.name
    fname = output_path.name
    # Parent directory of the inference folder
    base_dir = Path(__file__).resolve().parent.parent
    output_file = base_dir / "outputs" / split / fname
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        custom_json_dump_horizontal_predictions(grouped_results, f)

    # Convert sentence_id to string in analysis output as well
    results_str = []
    for r in results:
        r2 = r.copy()
        r2["sentence_id"] = str(r2["sentence_id"])
        results_str.append(r2)

    analysis_file = base_dir / "analysis" / split / fname
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_file, "w") as f:
        json.dump(results_str, f, indent=2)

    print(f"[DONE] Subtask 2 inference finished.\nSubmission file: {output_file}\nAnalysis file: {analysis_file}")


if __name__ == "__main__":
    main()
