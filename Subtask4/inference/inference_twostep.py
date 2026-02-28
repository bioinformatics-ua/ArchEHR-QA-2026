import argparse
import json
import re
from pathlib import Path

from dataloader import ArchEHRSubtask4DataLoader
from providers.cloud import CloudProvider
from providers.local import LocalProvider


# ── Hedging phrases that signal clinical knowledge sentences ──────────────────
# Answer sentences containing these are likely based on general medical knowledge
# rather than the note, so we force evidence_id = [] as post-processing.
CLINICAL_KNOWLEDGE_PHRASES = [
    "from my clinical knowledge",
    "from clinical knowledge",
    "based on clinical knowledge",
    "it is possible that",
    "may occur if",
    "can cause",
    "should not be concerned",
    "this should not be",
    "once antibiotics are completed",
    "should resolve",
]

def parse_args():
    parser = argparse.ArgumentParser(description="ArchEHR-QA Subtask 4 Two-Step Inference")

    # Data & I/O
    parser.add_argument("--xml-file", type=str, required=True)
    parser.add_argument("--qa-key-file", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--prompt-index", type=int, default=4)
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

    # Two-step control
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Skip second-pass verification (runs like standard inference.py)",
    )
    parser.add_argument(
        "--no-clinical-knowledge-filter",
        action="store_true",
        help="Skip the clinical knowledge heuristic post-processing filter",
    )

    return parser.parse_args()


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


def extract_id_list_from_text(text: str) -> list[str]:
    """
    Extract a JSON list of evidence IDs from second-pass output.
    Handles both full JSON objects and bare lists.
    """
    text = text.strip()

    # Try bare list first: ["1", "2"] or []
    match = re.search(r'\[([^\[\]]*)\]', text)
    if match:
        inner = match.group(0)
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    # Try full JSON object
    parsed = extract_json_from_text(text)
    if isinstance(parsed, dict):
        ids = parsed.get("additional_evidence_id") or parsed.get("evidence_id") or []
        return [str(x) for x in ids]

    return []


def is_clinical_knowledge_sentence(text: str) -> bool:
    """
    Heuristic: return True if the answer sentence appears to be based on
    general clinical knowledge rather than the note content.
    These sentences should have evidence_id = [].
    """
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in CLINICAL_KNOWLEDGE_PHRASES)


def build_second_pass_prompt(
    case: dict,
    answer_id: str,
    answer_text: str,
    current_evidence_ids: list[str],
) -> str:
    """
    Build a focused second-pass prompt for a single answer sentence
    that already has at least one citation. Asks the model to check
    if any additional note sentences were missed.
    """
    # Get the current citations as a readable list
    current_str = ", ".join(f"[{e}]" for e in current_evidence_ids)

    prompt = f"""You are a clinical evidence alignment expert performing a verification check.

An answer sentence has already been aligned to some clinical note sentences.
Your job is to check if any ADDITIONAL note sentences were missed.

### Rules
1. Only add note sentence IDs that EXPLICITLY state the same facts as the answer sentence.
2. Do NOT add sentences that are merely related or provide background context.
3. Do NOT remove or change the existing citations — only identify additions.
4. If no additional sentences are needed, return an empty list [].

### Output Format
Return ONLY a valid JSON list of additional evidence IDs. No preamble or markdown.
Example: ["3", "7"]
Example (nothing to add): []

### Input
Answer Sentence [{answer_id}]: {answer_text}

Already cited note sentences: {current_str}

All Clinical Note Sentences:
{case['note_sentences']}

Additional evidence IDs (or [] if none):"""

    return prompt


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

    # ── PASS 1: Standard inference ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("PASS 1: Standard inference")
    print("=" * 50)

    prompts = [provider.build_prompt(prompt_template, case) for case in cases]

    print("\nDEBUG: FULL PROMPT FOR CASE 0:")
    print(prompts[0])
    print("=" * 50 + "\n")

    print("Running pass 1 batch generation...")
    raw_outputs_p1 = provider.batch_generate(prompts)

    print("\nDEBUG: RAW LLM OUTPUT FOR CASE 0 (Pass 1):")
    print(repr(raw_outputs_p1[0]))
    print("=" * 50 + "\n")

    # Parse pass 1 results
    print("Parsing pass 1 outputs...")
    pass1_results = []
    for case, raw_output in zip(cases, raw_outputs_p1):
        parsed_json = extract_json_from_text(raw_output)
        is_valid = isinstance(parsed_json, dict) and "prediction" in parsed_json

        predictions = []
        if is_valid:
            for p in parsed_json["prediction"]:
                predictions.append({
                    "answer_id": str(p.get("answer_id", "")),
                    "evidence_id": [str(e) for e in p.get("evidence_id", [])],
                })
        else:
            print(f"Warning: Invalid/Empty pass 1 output for Case {case['case_id']}. Using fallback.")
            for ans in case["raw_answer_sentences"]:
                predictions.append({
                    "answer_id": str(ans.get("answer_id")),
                    "evidence_id": [],
                })

        pass1_results.append({
            "case_id": case["case_id"],
            "prediction": predictions,
            "_raw_answer_sentences": case["raw_answer_sentences"],
        })

    # ── CLINICAL KNOWLEDGE FILTER ─────────────────────────────────────────────
    if not args.no_clinical_knowledge_filter:
        print("\n" + "=" * 50)
        print("POST-PROCESSING: Clinical knowledge heuristic filter")
        print("=" * 50)
        filtered_count = 0
        # Build a lookup from case_id to raw answer sentences for text access
        case_ans_lookup = {c["case_id"]: {s["answer_id"]: s["text"] for s in c["_raw_answer_sentences"]} for c in pass1_results}
        for result in pass1_results:
            cid = result["case_id"]
            for pred in result["prediction"]:
                ans_text = case_ans_lookup[cid].get(pred["answer_id"], "")
                if pred["evidence_id"] and is_clinical_knowledge_sentence(ans_text):
                    print(f"  Case {cid} a{pred['answer_id']}: forcing [] (clinical knowledge) | '{ans_text[:70]}'")
                    pred["evidence_id"] = []
                    filtered_count += 1
        print(f"Filtered {filtered_count} sentences to [] via clinical knowledge heuristic.")

    # ── PASS 2: Verification for missed citations ─────────────────────────────
    if args.no_second_pass:
        print("\nSecond pass skipped (--no-second-pass flag set).")
        final_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in pass1_results]
    else:
        print("\n" + "=" * 50)
        print("PASS 2: Verification for missed citations")
        print("=" * 50)

        # Build case lookup for note sentences
        case_lookup = {c["case_id"]: c for c in cases}

        # Collect all second-pass prompts, tracking (result_idx, pred_idx)
        second_pass_items = []  # (result_idx, pred_idx, answer_id, answer_text, current_ids)

        case_ans_lookup = {
            c["case_id"]: {s["answer_id"]: s["text"] for s in c["_raw_answer_sentences"]}
            for c in pass1_results
        }

        for r_idx, result in enumerate(pass1_results):
            cid = result["case_id"]
            for p_idx, pred in enumerate(result["prediction"]):
                # Only run second pass on sentences that already have ≥1 citation
                if pred["evidence_id"]:
                    ans_text = case_ans_lookup[cid].get(pred["answer_id"], "")
                    second_pass_items.append((r_idx, p_idx, pred["answer_id"], ans_text, pred["evidence_id"]))

        print(f"Second pass targets: {len(second_pass_items)} answer sentences with existing citations")

        # Build second-pass prompts
        second_pass_prompts = []
        for r_idx, p_idx, answer_id, answer_text, current_ids in second_pass_items:
            cid = pass1_results[r_idx]["case_id"]
            case = case_lookup[cid]
            raw_prompt = build_second_pass_prompt(case, answer_id, answer_text, current_ids)
            # Use provider's chat template
            second_pass_prompts.append(
                provider.build_prompt(raw_prompt, {"case_id": cid, "patient_question": "", "clinician_question": "", "note_sentences": "", "answer_sentences": ""})
                if False  # We bypass build_prompt since we have a raw string
                else provider.tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                ) if args.inference_mode == "local"
                else [{"role": "user", "content": raw_prompt}]
            )

        print("Running pass 2 batch generation...")
        raw_outputs_p2 = provider.batch_generate(second_pass_prompts)

        print("\nDEBUG: RAW LLM OUTPUT FOR PASS 2 ITEM 0:")
        print(repr(raw_outputs_p2[0]))
        print("=" * 50 + "\n")

        # Merge second-pass results back
        added_total = 0
        for i, (r_idx, p_idx, answer_id, answer_text, current_ids) in enumerate(second_pass_items):
            additional_ids = extract_id_list_from_text(raw_outputs_p2[i])

            # Only keep IDs that are genuinely new and not duplicates
            existing = set(current_ids)
            new_ids = [eid for eid in additional_ids if eid not in existing]

            if new_ids:
                cid = pass1_results[r_idx]["case_id"]
                print(f"  Case {cid} a{answer_id}: adding {new_ids} to existing {current_ids}")
                pass1_results[r_idx]["prediction"][p_idx]["evidence_id"].extend(new_ids)
                added_total += len(new_ids)

        print(f"\nPass 2 added {added_total} new citation(s) across all cases.")

        # Strip internal keys before saving
        final_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in pass1_results]

    # ── Save output ───────────────────────────────────────────────────────────
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nSuccessfully saved results to {output_path}")


if __name__ == "__main__":
    main()