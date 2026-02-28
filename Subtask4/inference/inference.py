import argparse
import json
import re
from pathlib import Path

from dataloader import ArchEHRSubtask4DataLoader
from providers.cloud import CloudProvider
from providers.local import LocalProvider

def parse_args():
    parser = argparse.ArgumentParser(description="ArchEHR-QA Subtask 4 Inference")
    
    # Data & I/O
    parser.add_argument("--xml-file", type=str, required=True, help="Path to archehr-qa.xml")
    parser.add_argument("--qa-key-file", type=str, required=True, help="Path to JSON key file")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--prompt-index", type=int, default=1, help="Index of the prompt to use")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--debug-first-n", type=int, default=None, help="Run only on first N cases")

    # Model & Mode
    parser.add_argument("--inference-mode", type=str, choices=["local", "cloud"], required=True)
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    
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
    
    # 1. Load the prompt template
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompt_template = prompts_data[str(args.prompt_index)]
    print(f"Loaded prompt index {args.prompt_index}")

    # 2. Load the data using our finalized dataloader
    dataloader = ArchEHRSubtask4DataLoader(
        xml_path=args.xml_file, 
        json_path=args.qa_key_file
    )
    cases = dataloader.load()
    
    if args.debug_first_n:
        cases = cases[:args.debug_first_n]
        print(f"DEBUG MODE: Running only on first {args.debug_first_n} cases.")
    else:
        print(f"Loaded {len(cases)} cases for inference.")

    # 3. Initialize Provider
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

    # 4. Build Prompts
    print("Building prompts...")
    # The provider handles injecting `note_sentences` and `answer_sentences` strings 
    # directly into the prompt template since their dictionary keys match the placeholders perfectly.
    prompts = [provider.build_prompt(prompt_template, case) for case in cases]

    # --- DEBUG: Look at the exact prompt being sent to the model ---
    print("\n" + "="*50)
    print("DEBUG: FULL PROMPT FOR CASE 0:")
    print(prompts[0])
    print("="*50 + "\n")

    # 5. Run Generation
    print("Running batch generation...")
    raw_outputs = provider.batch_generate(prompts)

    # --- DEBUG: Look at the raw text the LLM generated ---
    print("\n" + "="*50)
    print("DEBUG: RAW LLM OUTPUT FOR CASE 0:")
    print(repr(raw_outputs[0])) # using repr() to clearly see empty strings or hidden newlines
    print("="*50 + "\n")
    
    # 6. Parse and format output safely
    print("Parsing outputs...")
    results = []
    for case, raw_output in zip(cases, raw_outputs):
        parsed_json = extract_json_from_text(raw_output)
        
        # Check if the LLM gave us valid keys matching the submission format
        is_valid = isinstance(parsed_json, dict) and "prediction" in parsed_json
        
        predictions = []

        if is_valid:
            # Trust but verify: loop through what the LLM gave us and strictly enforce data types
            for p in parsed_json["prediction"]:
                predictions.append({
                    "answer_id": str(p.get("answer_id", "")),
                    "evidence_id": [str(e) for e in p.get("evidence_id", [])]
                })
        else:
            # FALLBACK: The LLM failed (blank output, refused to answer, bad JSON)
            print(f"Warning: Invalid/Empty output for Case {case['case_id']}. Using fallback.")
            # Build a perfectly structured empty response using the raw dictionary from our dataloader
            for ans in case["raw_answer_sentences"]:
                predictions.append({
                    "answer_id": str(ans.get("answer_id")),
                    "evidence_id": []
                })

        # Append final submission object for this case
        results.append({
            "case_id": case["case_id"],
            "prediction": predictions
        })

    # 7. Save to output file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully saved results to {output_path}")

if __name__ == "__main__":
    main()