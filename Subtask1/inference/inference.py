import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer # Import the tokenizer

# --- XML Data Loader ---
class ArchEHRDataLoader:
    def __init__(self, xml_path):
        self.xml_path = xml_path

    def load(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        data = []
        for case in root.findall('case'):
            case_id = case.get('id')
            # Try both field names (test uses patient_narrative, dev uses clinician_question)
            patient_text = case.findtext('patient_narrative', '').strip()
            
            if patient_text:
                data.append({
                    "case_id": case_id,
                    "clinician_question": patient_text
                })
        return data

def main():
    # --- 1. The Settings Menu (Arguments) ---
    parser = argparse.ArgumentParser(description="Batch inference with VLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B", 
        help="Hugging Face model to use."
    )
    parser.add_argument("--xml-file", type=str, required=True, help="Path to the XML file with patient narratives.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the prompt template JSONL file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSON file for results.")
    args = parser.parse_args()

    # --- 1. Load the tokenizer for manual decoding ---
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Tokenizer loaded.")

    # --- 2. Load XML Cases ---
    print(f"Loading XML data from {args.xml_file}...")
    loader = ArchEHRDataLoader(args.xml_file)
    xml_cases = loader.load()
    print(f"Loaded {len(xml_cases)} cases from XML")

    # --- 3. Load Prompt Template ---
    print(f"Loading prompt template from {args.prompt_file}...")
    with open(args.prompt_file, 'r') as f:
        prompt_template_data = json.loads(f.readline())
        prompt_template = prompt_template_data["text"]
    print("Prompt template loaded.")

    # --- 4. Build Prompts ---
    prompts = []
    prompt_data = []
    for case in xml_cases:
        # Replace the placeholder with the actual patient narrative
        filled_prompt = prompt_template.replace("{PATIENT_NARRATIVE}", case["clinician_question"])
        
        # Apply the chat template
        messages = [{"role": "user", "content": filled_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True
        )
        prompts.append(formatted_prompt)
        prompt_data.append(case)
    
    print(f"Built {len(prompts)} prompts.")

    # --- 3. Initialize VLLM ---
    print("Initializing VLLM Engine...")

    # max_model_len: The limit of how much text it can remember (8192 words/tokens is usually safe)
    # tensor_parallel_size: How many GPUs to use (1 for your current setup)
    # enforce_eager=True: Disables torch.compile to avoid compilation errors
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=8192, enforce_eager=True)

    # SamplingParams controls "Creativity":
    # temperature=0.7: A balance between creative and precise.
    # max_tokens=1024: Stop generating if the answer gets longer than this.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
    print("VLLM engine initialized.")

    # --- 4. Run Batch Inference ---
    print("Starting batch generation...")
    outputs = llm.generate(prompts, sampling_params)
    print("Batch generation complete.")

    # --- 5. Save Results in gold.json format ---
    print(f"Saving results to {args.output_file}...")
    results = []
    
    for i, output in enumerate(outputs):
        original_case = prompt_data[i]
        
        # Get the generated text without special tokens for parsing
        generated_text = output.outputs[0].text.strip()
        
        # Extract the query from the JSON in the generated text
        # Look for the last JSON object in the response
        try:
            # Find JSON pattern {"query": "..."} or similar
            json_match = re.search(r'\{[^{}]*"query"[^{}]*\}', generated_text)
            if json_match:
                query_json = json.loads(json_match.group())
                prediction = query_json.get("query", "")
            else:
                # Fallback: use the whole generated text
                prediction = generated_text
        except:
            # If parsing fails, use the generated text as is
            prediction = generated_text
        
        result = {
            "case_id": original_case["case_id"],
            "prediction": prediction
        }
        results.append(result)
    
    # Save as JSON array (not JSONL)
    output_dir = "outputs"
    
    base_name = os.path.basename(args.output_file)
    output_path = os.path.join(output_dir, base_name)
    
    print(f"\n>>> Saving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path} in gold.json format")

if __name__ == "__main__":
    main()