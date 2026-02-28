import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import os

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
        description="Subtask 4: Answer-Evidence Alignment"
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
        if sent_block:
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
# BUILD CHAT MESSAGES
# -------------------------------------------------------

def build_messages(system_prompt, user_template, case_id, case_data, answer_sents):
    # Implementing your teacher's semantic tagging trick for note sentences
    numbered_notes = "\n".join(
        f'[s{s["sentence_id"]}] {s["text"]}' for s in case_data["note_sentences"]
    )
    
    # You should do the exact same trick for answer sentences!
    numbered_answers = "\n".join(
        f'[a{s["answer_id"]}] {s["text"]}' for s in answer_sents
    )

    user_content = user_template.format(
        case_id=case_id,
        patient_question=case_data["patient_question"],
        clinician_question=case_data["clinician_question"],
        note_sentences=numbered_notes,
        answer_sentences=numbered_answers,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# -------------------------------------------------------
# EXTRACT JSON SAFELY
# -------------------------------------------------------

import re

def extract_json(text, case_id="Unknown"):
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        print(f"[!] Case {case_id}: No JSON brackets found in response.")
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        print(f"[!] Case {case_id} JSON Parse Error: {e}")
        print(f"[!] Broken text snippet: {text[start:start+100]}...")
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
        raise IndexError(
            f"Prompt index {args.prompt_index} out of range "
            f"(file has {len(prompts)} prompts, max index is {len(prompts) - 1})"
        )

    prompt_config = prompts[args.prompt_index]
    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]

    # --- Init provider ---
    provider = build_provider(args)
    print(f"Provider: {args.inference_mode} | Model: {args.model}")

    # --- Load data ---
    cases = load_xml_cases(args.xml_file)
    answers = load_answers(args.qa_key_file)

    # --- Ensure output directory exists ---
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # --- Run inference ---
    # --- Build all messages ---
    case_ids = []
    all_messages = []

    for case_id in cases:
        if case_id not in answers:
            continue
        if args.only_case_id and case_id != args.only_case_id:
            continue

        case_data = cases[case_id]
        answer_sents = answers[case_id]

        messages = build_messages(
            system_prompt, user_template, case_id, case_data, answer_sents
        )
        case_ids.append(case_id)
        all_messages.append(messages)

    # --- Batch inference ---
    print(f"Running inference on {len(case_ids)} cases...")
    responses = provider.batch_generate(all_messages)

    # --- Parse responses ---
    submission = []

    for case_id, response in zip(case_ids, responses):
        if args.debug_first_n > 0:
            print("\n==============================")
            print("CASE:", case_id)
            print("RAW RESPONSE:")
            print(response)
            print("==============================\n")
            args.debug_first_n -= 1

        parsed = extract_json(response)
        prediction = []

        if parsed and "alignment" in parsed:
            for item in parsed["alignment"]:
                # Grab the ID and strip the 'a' prefix (turns "a1" back into "1")
                raw_answer_id = str(item.get("answer_sentence_id", ""))
                clean_answer_id = raw_answer_id.replace("a", "").strip()
                
                # Grab the evidence list and strip the 's' prefix (turns ["s2", "s5"] back into ["2", "5"])
                evidence_ids = item.get("evidence", [])
                clean_evidence_ids = [str(e).replace("s", "").strip() for e in evidence_ids]

                prediction.append({
                    "answer_id": clean_answer_id,
                    "evidence_id": clean_evidence_ids,
                })

        submission.append({
            "case_id": case_id,
            "prediction": prediction,
        })

        print(f"Finished case {case_id}")

    # --- Save ---
    with open(args.output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()