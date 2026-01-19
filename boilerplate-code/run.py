import os
import json
import time
import re
import argparse
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# --- 1. Custom Dataloader ---
class ArchEHRDataLoader:
    def __init__(self, key_path, mapping_path, xml_path):
        self.key_path = key_path
        self.mapping_path = mapping_path
        self.xml_path = xml_path

    def load(self):
        with open(self.key_path, 'r', encoding='utf-8') as f:
            keys = {item['case_id']: item for item in json.load(f)}
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            mappings = {item['case_id']: item for item in json.load(f)}
        
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        combined_data = []
        for case in root.findall('case'):
            case_id = case.get('id')
            ans_data = keys.get(case_id, {}).get('answers', [])
            
            relevance_labels = {ans['sentence_id']: ans['relevance'] for ans in ans_data}
            
            mapped_sentence_strings = []
            sentence_elements = case.find('note_excerpt_sentences').findall('sentence')
            for sent in sentence_elements:
                s_id = sent.get('id')
                text = sent.text.strip() if sent.text else ""
                mapped_sentence_strings.append(f"[s{s_id}] {text}")

            combined_data.append({
                "case_id": case_id,
                "clinician_question": case.findtext('clinician_question', '').strip(),
                "sentence_mapped_view": f"Document {case_id}: " + " ".join(mapped_sentence_strings),
                "ground_truth_labels": {
                    "essential_ids": [s_id for s_id, rel in relevance_labels.items() if rel == "essential"],
                    "supplementary_ids": [s_id for s_id, rel in relevance_labels.items() if rel == "supplementary"],
                    "not_relevant_ids": [s_id for s_id, rel in relevance_labels.items() if rel == "not-relevant"]
                },
                "clinician_ground_truth": keys.get(case_id, {}).get("clinician_answer")
            })
        return combined_data

# --- 2. Prompt Definition (Simplified for Reasoning Models) ---
SYSTEM_PROMPT = "You are a clinical auditor. You provide reasoning followed by a structured JSON audit."

AUDITOR_USER_PROMPT = """Analyze the EHR snippet to answer: {clinician_question}

Evaluation Criteria:
- Essential (2): Direct cause/required intervention.
- Supplementary (1): Safety/context results.
- Not Relevant (0): Administrative/unrelated data.

Task: Provide a sentence-by-sentence audit in the following JSON format:
{{
  "s1": {{"score": 0-2, "justification": "..."}},...
  "sn": {{"score": 0-2, "justification": "..."}}, 
}}

Context:
{sentence_mapped_view}
"""

# --- 3. Helper to split CoT and JSON ---
def parse_reasoning_output(text: str) -> Tuple[str, str]:
    """Extracts content inside <think> tags and the JSON outside them."""
    thinking = ""
    audit_json = text
    
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        audit_json = text.split('</think>')[-1].strip()
    
    # Clean up potential markdown blocks in the remaining JSON
    audit_json = re.sub(r'```json\s*|\s*```', '', audit_json)
    return thinking, audit_json

# --- 4. Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="vLLM Clinical Auditor (Reasoning Model)")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--base-dir", type=str, default="./dataset_1.3/dev/")
    parser.add_argument("--output-file", type=str, default="results/audit_results.jsonl")
    args = parser.parse_args()

    # Model Setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    llm = LLM(model=args.model_id, max_model_len=4096, gpu_memory_utilization=0.9, trust_remote_code=True) #ma model
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.6) # Reasoning models usually need higher temp

    # Load Data
    loader = ArchEHRDataLoader(
        os.path.join(args.base_dir, 'archehr-qa_key.json'),
        os.path.join(args.base_dir, 'archehr-qa_mapping.json'),
        os.path.join(args.base_dir, 'archehr-qa.xml')
    )
    all_cases = loader.load()
    
    # Sharding
    my_cases = [c for i, c in enumerate(all_cases) if i % args.world_size == args.rank]
    
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, 
         {"role": "user", "content": AUDITOR_USER_PROMPT.format(**c)}],
        tokenize=False, add_generation_prompt=True) for c in my_cases]

    if prompts:
        outputs = llm.generate(prompts, sampling_params)
        
        with open(f"{args.output_file}.part_{args.rank}", "a", encoding="utf-8") as f:
            for i, output in enumerate(outputs):
                case_data = my_cases[i]
                raw_text = output.outputs[0].text
                
                # Split the auto-generated thinking from the JSON
                thinking, cleaned_json_text = parse_reasoning_output(raw_text)
                
                result = {
                    "case_id": case_data["case_id"],
                    "thinking": thinking,
                    "parsed_audit": cleaned_json_text,
                    "ground_truth_labels": case_data["ground_truth_labels"],
                    "clinician_answer_reference": case_data["clinician_ground_truth"],
                    "raw_output": raw_text
                }
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()