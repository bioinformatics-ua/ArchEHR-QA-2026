import json
import argparse
from pathlib import Path

from dataloader import ArchEHRSubtask3DataLoader


def main():
    parser = argparse.ArgumentParser("Subtask 3: Answer Generation inference")
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
        help="Max tokens to generate per answer",
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
    cases = ArchEHRSubtask3DataLoader(args.xml_file).load()
    print(f"Loaded {len(cases)} cases.")

    # --- Load prompt template ---
    with open(args.prompt_file) as f:
        prompt_dict = json.load(f)
    prompt_template = prompt_dict[str(args.prompt_index)]

    # --- Build prompts ---
    prompts_with_cases = []
    for case in cases:
        sentences_text = "\n".join(
            f"{s['sentence_id']}: {s['text']}" for s in case["sentences"]
        )
        payload = {
            "patient_question": case["patient_question"],
            "clinician_question": case["clinician_question"],
            "sentences": sentences_text,
            "note_excerpt": case["note_excerpt"],
            "case_id": case["case_id"],
        }
        prompt = provider.build_prompt(prompt_template, payload)
        prompts_with_cases.append((case, prompt))

    print(f"Built {len(prompts_with_cases)} prompts. Running inference...")

    # --- Batch inference ---
    all_prompts = [prompt for _, prompt in prompts_with_cases]
    outputs = provider.batch_generate(all_prompts)

    # --- Parse and build submission ---
    MAX_WORDS = 75
    results = []
    for (case, _), raw_output in zip(prompts_with_cases, outputs):
        answer = provider.parse_response(raw_output)

        if not answer:
            print(f"[case {case['case_id']}]: Empty output, inserting fallback.")
            answer = "Insufficient information provided in the note excerpt."

        # Enforce 75-word limit
        words = answer.split()
        if len(words) > MAX_WORDS:
            print(
                f"[case {case['case_id']}]: {len(words)} words, "
                f"truncating to {MAX_WORDS}."
            )
            answer = " ".join(words[:MAX_WORDS])

        results.append(
            {
                "case_id": str(case["case_id"]),
                "prediction": answer,
            }
        )

    # --- Write submission JSON ---
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Write analysis JSON (full raw outputs for debugging) ---
    base_dir = Path(__file__).resolve().parent.parent
    split = output_path.parent.name
    fname = output_path.name
    analysis_file = base_dir / "analysis" / split / fname
    analysis_file.parent.mkdir(parents=True, exist_ok=True)

    analysis_data = []
    for (case, _), raw_output in zip(prompts_with_cases, outputs):
        analysis_data.append(
            {
                "case_id": str(case["case_id"]),
                "raw_output": raw_output,
                "parsed_answer": provider.parse_response(raw_output),
                "word_count": len(provider.parse_response(raw_output).split()),
            }
        )
    with open(analysis_file, "w") as f:
        json.dump(analysis_data, f, indent=2)

    print(
        f"[DONE] Subtask 3 inference finished.\n"
        f"Submission: {output_path}\n"
        f"Analysis:   {analysis_file}"
    )


if __name__ == "__main__":
    main()
