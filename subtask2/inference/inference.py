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

    args = parser.parse_args()

    # --- Provider (IMPORT LAZY, SEM vllm NO CLOUD) ---
    if args.inference_mode == "cloud":
        from providers.cloud import CloudProvider
        provider = CloudProvider(args.model)
    else:
        from providers.local import LocalProvider
        provider = LocalProvider(args.model)

    # --- Load data ---
    loader = ArchEHRSubtask2DataLoader(args.xml_file)
    cases = loader.load()

    # --- Load prompt ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)
    prompt_template = prompt_dict[str(args.prompt_index)]

    results = []

    for case in cases:
        for sent in case["sentences"]:
            payload = {
                "clinician_question": case["clinician_question"],
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

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("[DONE] Subtask 2 inference finished")


if __name__ == "__main__":
    main()
