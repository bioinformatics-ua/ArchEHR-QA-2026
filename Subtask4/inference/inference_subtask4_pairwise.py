import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import os
import re
from collections import defaultdict

from dotenv import load_dotenv

# -------------------------------------------------------
# LOAD .env FILE (must happen before provider init)
# -------------------------------------------------------

def load_env():
    """Load .env from SLURM_SUBMIT_DIR or cwd."""
    submit_dir = os.getenv("SLURM_SUBMIT_DIR")
    env_path = Path(submit_dir) / ".env" if submit_dir else Path.cwd() / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"WARNING: .env not found at {env_path}")

# -------------------------------------------------------
# ARGUMENTS
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Subtask 4: Answer-Evidence Alignment (Pairwise Batching)"
    )

    # --- I/O ---
    parser.add_argument("--xml-file", required=True,
                        help="Path to archehr-qa.xml")
    parser.add_argument("--qa-key-file", required=True,
                        help="Path to archehr-qa_key.json (contains answer sentences)")
    parser.add_argument("--prompt-file", required=True,
                        help="Path to prompt_subtask4.json")
    parser.add_argument("--prompt-index", type=int, required=True,
                        help="Index into prompt file")
    parser.add_argument("--output-file", required=True,
                        help="Path to write submission JSON")

    # --- Provider ---
    parser.add_argument("--inference-mode", choices=["local", "cloud"],
                        default="cloud",
                        help="Use local (vLLM) or cloud (OpenRouter) provider")
    parser.add_argument("--model", required=True,
                        help="Model name (HF repo for local, model string for cloud)")

    # --- Sampling (shared) ---
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)

    # --- Local-only (vLLM) ---
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    # --- Debug ---
    parser.add_argument("--debug-first-n", type=int, default=0,
                        help="Print raw LLM output for first N cases")
    parser.add_argument("--only-case-id", type=str, default=None,
                        help="Run only a single case (for debugging)")

    return parser.parse_args()

# -------------------------------------------------------
# PROVIDER FACTORY
# -------------------------------------------------------

def build_provider(args):
    """Instantiate the correct provider based on --inference-mode."""
    if args.inference_mode == "cloud":
        from providers.cloud import CloudProvider
        return CloudProvider(
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    else:
        from providers.local import LocalProvider
        return LocalProvider(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
        )

# -------------------------------------------------------
# LOAD XML CASES
# -------------------------------------------------------

def load_xml_cases(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases = {}

    for case in root.findall("case"):
        case_id = case.get("id")

        patient_q = case.findtext("patient_narrative", "").strip()
        clinician_q = case.findtext("clinician_question", "").strip()

        sentences = []
        sent_block = case.find("note_excerpt_sentences")
        if sent_block is not None:  # <-- Fixed the DeprecationWarning here!
            for s in sent_block.findall("sentence"):
                sentences.append({
                    "sentence_id": s.get("id"),
                    "text": (s.text or "").strip().replace("\n", " ")
                })

        cases[case_id] = {
            "patient_question": patient_q,
            "clinician_question": clinician_q,
            "note_sentences": sentences,
        }

    return cases

# -------------------------------------------------------
# LOAD ANSWER SENTENCES
# -------------------------------------------------------

def load_answers(key_path):
    with open(key_path) as f:
        data = json.load(f)

    answers = {}
    for case in data:
        case_id = case["case_id"]
        answer_sents = []
        for s in case["clinician_answer_sentences"]:
            answer_sents.append({
                "answer_id": s["id"],
                "text": s["text"],
            })
        answers[case_id] = answer_sents

    return answers

# -------------------------------------------------------
# BUILD CHAT MESSAGES (PAIRWISE)
# -------------------------------------------------------

def build_messages(system_prompt, user_template, case_id, case_data, target_answer_id, target_answer_text):
    # Apply [s1] tags to note sentences
    numbered_notes = "\n".join(
        f'[s{s["sentence_id"]}] {s["text"]}' for s in case_data["note_sentences"]
    )
    
    # Inject ONLY the target answer sentence into the prompt
    user_content = user_template.format(
        case_id=case_id,
        patient_question=case_data.get("patient_question", ""),
        clinician_question=case_data.get("clinician_question", ""),
        note_sentences=numbered_notes,
        answer_id=target_answer_id,
        answer_text=target_answer_text,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

# -------------------------------------------------------
# EXTRACT JSON SAFELY
# -------------------------------------------------------

def extract_json(text):
    # Aggressively strip markdown code fences before searching for brackets
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1

    if start == -1 or end == 0:
        return None
    try:
        return json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    load_env()
    args = parse_args()

    # --- Load prompt ---
    with open(args.prompt_file) as f:
        prompts = json.load(f)

    if args.prompt_index >= len(prompts):
        raise IndexError(f"Prompt index {args.prompt_index} out of range.")

    prompt_config = prompts[args.prompt_index]
    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]

    # --- Init provider ---
    provider = build_provider(args)
    print(f"Provider: {args.inference_mode} | Model: {args.model}")

    # --- Load data ---
    cases = load_xml_cases(args.xml_file)
    answers = load_answers(args.qa_key_file)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # FLATTEN QUERIES FOR PAIRWISE BATCHING
    # ---------------------------------------------------
    flat_queries = []

    for case_id in cases:
        if case_id not in answers:
            continue
        if args.only_case_id and case_id != args.only_case_id:
            continue

        case_data = cases[case_id]
        answer_sents = answers[case_id]

        # Create a separate prompt for EVERY answer sentence
        for ans in answer_sents:
            ans_id = str(ans["answer_id"])
            messages = build_messages(
                system_prompt, user_template, case_id, case_data, ans_id, ans["text"]
            )
            
            flat_queries.append({
                "case_id": case_id,
                "answer_id": ans_id,
                "text": ans["text"],
                "messages": messages
            })

    # --- Batch inference ---
    print(f"Running pairwise inference on {len(flat_queries)} total answer sentences...")
    all_messages = [q["messages"] for q in flat_queries]
    responses = provider.batch_generate(all_messages)

    # ---------------------------------------------------
    # RECONSTRUCT RESULTS BY CASE
    # ---------------------------------------------------
    case_predictions = defaultdict(list)

    for query, response in zip(flat_queries, responses):
        case_id = query["case_id"]
        ans_id = query["answer_id"]

        if args.debug_first_n > 0:
            print("\n==============================")
            print(f"CASE: {case_id} | ANSWER: {ans_id}")
            print(f"TEXT: {query['text']}")
            print("RAW RESPONSE:")
            print(response)
            print("==============================\n")
            args.debug_first_n -= 1

        parsed = extract_json(response)
        clean_evidence_ids = []

        if parsed and "evidence" in parsed:
            evidence_ids = parsed.get("evidence", [])
            if isinstance(evidence_ids, list):
                # Turn ['s2', 's5'] back into ['2', '5'] for the Codabench grader
                clean_evidence_ids = [str(e).replace("s", "").strip() for e in evidence_ids]

        case_predictions[case_id].append({
            "answer_id": ans_id,
            "evidence_id": clean_evidence_ids,
        })

    # --- Build final submission format ---
    submission = []
    # Loop through the original cases dictionary to maintain order
    for case_id in cases:
        if case_id in case_predictions:
            submission.append({
                "case_id": case_id,
                "prediction": case_predictions[case_id]
            })

    # --- Save ---
    with open(args.output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"Saved: {args.output_file}")

if __name__ == "__main__":
    main()