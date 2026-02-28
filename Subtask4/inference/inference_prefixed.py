import argparse
import json
import re
from pathlib import Path

from dataloader import ArchEHRSubtask4DataLoader
from providers.cloud import CloudProvider
from providers.local import LocalProvider


def parse_args():
    parser = argparse.ArgumentParser(description="ArchEHR-QA Subtask 4 Inference with N/A ID prefixes")

    # Data & I/O
    parser.add_argument("--xml-file", type=str, required=True)
    parser.add_argument("--qa-key-file", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--prompt-index", type=int, default=3)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--debug-first-n", type=int, default=None)

    # Model & Mode
    parser.add_argument("--inference-mode", type=str, choices=["local", "cloud"], required=True)
    parser.add_argument("--model", type=str, required=True)

    # Engine parameters (Local only)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    return parser.parse_args()


def add_prefixes(case: dict) -> dict:
    """
    Return a copy of the case with N-prefixed note sentences and A-prefixed
    answer sentences. The original case dict is not modified.

    Transforms:
        note_sentences:    "[1] text..."   ->  "[N1] text..."
        answer_sentences:  "[1] text..."   ->  "[A1] text..."
    """
    # Note sentences: replace leading [digits] with [Ndigits]
    note_prefixed = re.sub(
        r'^\[(\d+)\]',
        r'[N\1]',
        case['note_sentences'],
        flags=re.MULTILINE
    )
    # Answer sentences: replace leading [digits] with [Adigits]
    ans_prefixed = re.sub(
        r'^\[(\d+)\]',
        r'[A\1]',
        case['answer_sentences'],
        flags=re.MULTILINE
    )
    return {
        **case,
        'note_sentences': note_prefixed,
        'answer_sentences': ans_prefixed,
    }


def strip_n_prefix(evidence_ids: list[str]) -> list[str]:
    """
    Strip the N prefix from evidence IDs before saving.
    "N1" -> "1", "N12" -> "12". Passes through IDs without prefix unchanged.
    """
    return [re.sub(r'^N(\d+)$', r'\1', eid) for eid in evidence_ids]


def extract_json_from_text(text: str) -> dict:
    """Safely extracts a JSON dictionary from a raw LLM string."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def main():
    args = parse_args()

    # 1. Load prompt template
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompt_template = prompts_data[str(args.prompt_index)]
    print(f"Loaded prompt index {args.prompt_index}")

    # 2. Load data
    dataloader = ArchEHRSubtask4DataLoader(
        xml_path=args.xml_file,
        json_path=args.qa_key_file,
    )
    cases = dataloader.load()

    if args.debug_first_n:
        cases = cases[:args.debug_first_n]
        print(f"DEBUG MODE: Running only on first {args.debug_first_n} cases.")
    else:
        print(f"Loaded {len(cases)} cases for inference.")

    # 3. Initialize provider
    if args.inference_mode == "cloud":
        print(f"Initializing Cloud Provider with model: {args.model}")
        provider = CloudProvider(
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    else:
        print(f"Initializing Local Provider with model: {args.model}")
        provider = LocalProvider(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
        )

    # 4. Apply N/A prefixes and build prompts
    print("Applying N/A prefixes to note and answer sentences...")
    prefixed_cases = [add_prefixes(case) for case in cases]

    print("Building prompts...")
    prompts = [provider.build_prompt(prompt_template, case) for case in prefixed_cases]

    # DEBUG
    print("\n" + "=" * 50)
    print("DEBUG: FULL PROMPT FOR CASE 0:")
    print(prompts[0])
    print("=" * 50 + "\n")

    # 5. Run generation
    print("Running batch generation...")
    raw_outputs = provider.batch_generate(prompts)

    print("\n" + "=" * 50)
    print("DEBUG: RAW LLM OUTPUT FOR CASE 0:")
    print(repr(raw_outputs[0]))
    print("=" * 50 + "\n")

    # 6. Parse outputs and strip N prefix from evidence IDs before saving
    print("Parsing outputs...")
    results = []
    for case, raw_output in zip(cases, raw_outputs):
        parsed_json = extract_json_from_text(raw_output)
        is_valid = isinstance(parsed_json, dict) and "prediction" in parsed_json

        predictions = []
        if is_valid:
            for p in parsed_json["prediction"]:
                raw_ids = [str(e) for e in p.get("evidence_id", [])]
                raw_answer_id = str(p.get("answer_id", ""))
                predictions.append({
                    # Strip A prefix so output matches standard submission format
                    "answer_id": re.sub(r'^A(\d+)$', r'\1', raw_answer_id),
                    # Strip N prefix so output matches standard submission format
                    "evidence_id": strip_n_prefix(raw_ids),
                })
        else:
            print(f"Warning: Invalid/Empty output for Case {case['case_id']}. Using fallback.")
            for ans in case["raw_answer_sentences"]:
                predictions.append({
                    "answer_id": str(ans.get("answer_id")),
                    "evidence_id": [],
                })

        results.append({
            "case_id": case["case_id"],
            "prediction": predictions,
        })

    # 7. Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Successfully saved results to {output_path}")


if __name__ == "__main__":
    main()