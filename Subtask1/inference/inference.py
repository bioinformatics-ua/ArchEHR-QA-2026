import orjson
import os
import re
from pathlib import Path
from typing import Annotated

import typer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI

from dataloader import ArchEHRDataLoader

app = typer.Typer()


@app.command()
def main(
    xml_file: Annotated[
        Path, typer.Option(help="Path to the XML file with patient narratives.")
    ],
    prompt_file: Annotated[
        Path, typer.Option(help="Path to the prompt template JSONL file.")
    ],
    prompt_index: Annotated[str, typer.Option(help="Prompt id.")],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file for results.")
    ],
    inference_mode: Annotated[
        str,
        typer.Option(
            help="Inference mode: 'local' for vLLM, 'openai' for OpenAI API, or 'groq' for Groq API."
        ),
    ] = "local",
    model: Annotated[
        str, typer.Option(help="Hugging Face model to use.")
    ] = "Qwen/Qwen3-8B",
    openai_api_key: Annotated[
        str, typer.Option(help="OpenAI API key (or set OPENAI_API_KEY env variable).")
    ] = None,
) -> None:
    """Batch inference with VLLM."""
    # --- 1. Setup based on inference mode ---
    if inference_mode == "openai":
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required via --openai-api-key or OPENAI_API_KEY env variable"
            )
        client = OpenAI(api_key=api_key)
        print(f"Using OpenAI API with model: {model}")
        tokenizer = None
    elif inference_mode == "groq":
        api_key = openai_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required via --openai-api-key or GROQ_API_KEY env variable"
            )
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        print(f"Using Groq API with model: {model}")
        tokenizer = None
    else:
        print(f"Loading tokenizer for {model}...")
        hf_token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)
        print("Tokenizer loaded.")
        client = None

    # --- 2. Load XML Cases ---
    print(f"Loading XML data from {xml_file}...")
    loader = ArchEHRDataLoader(xml_file)
    xml_cases = loader.load()
    print(f"Loaded {len(xml_cases)} cases from XML")

    # --- 3. Load Prompt Template ---
    print(f"Loading prompt template from {prompt_file}...")
    with open(prompt_file, "r") as f:
        prompt_dict = orjson.loads(f.read())
    print("Prompt template loaded.")

    prompt_template = prompt_dict[prompt_index]
    # --- 4. Build Prompts ---
    prompts = []
    prompt_data = []
    for case in xml_cases:
        # Replace the placeholder with the actual patient narrative

        # TODO: Needs to be more flexible to match different prompt formats
        filled_prompt = prompt_template.replace(
            "{PATIENT_NARRATIVE}", case["clinician_question"]
        )

        if inference_mode == "local":
            # Apply the chat template for local models
            messages = [{"role": "user", "content": filled_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(formatted_prompt)
        else:
            # For OpenAI and Groq, store as message format
            prompts.append([{"role": "user", "content": filled_prompt}])

        prompt_data.append(case)

    print(f"Built {len(prompts)} prompts.")

    # --- 3. Run Inference Based on Mode ---
    if inference_mode == "local":
        # --- Initialize VLLM ---
        print("Initializing VLLM Engine...")

        # max_model_len: The limit of how much text it can remember (8192 words/tokens is usually safe)
        # tensor_parallel_size: How many GPUs to use (1 for your current setup)
        # enforce_eager=True: Disables torch.compile to avoid compilation errors
        # gpu_memory_utilization: Fraction of GPU memory to use (lower = more memory for weights)
        llm = LLM(
            model=model,
            tensor_parallel_size=1,
            max_model_len=8192,
            enforce_eager=True,
            gpu_memory_utilization=0.85
        )

        # SamplingParams controls "Creativity":
        # temperature=0.7: A balance between creative and precise.
        # max_tokens=1024: Stop generating if the answer gets longer than this.
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        print("VLLM engine initialized.")

        # --- Run Batch Inference ---
        print("Starting batch generation...")
        outputs = llm.generate(prompts, sampling_params)
        print("Batch generation complete.")

        # Process vLLM outputs
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
                    query_json = orjson.loads(json_match.group())
                    prediction = query_json.get("query", "")
                else:
                    # Fallback: use the whole generated text
                    prediction = generated_text
            except:
                # If parsing fails, use the generated text as is
                prediction = generated_text

            result = {"case_id": original_case["case_id"], "prediction": prediction}
            results.append(result)

    else:  # OpenAI or Groq mode
        print(f"Starting {inference_mode.upper()} API inference...")
        results = []

        for i, messages in enumerate(prompts):
            original_case = prompt_data[i]
            print(f"Processing case {i + 1}/{len(prompts)}: {original_case['case_id']}")

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )

                generated_text = response.choices[0].message.content.strip()

                # Extract the query from the JSON in the generated text
                try:
                    # Find JSON pattern {"query": "..."} or similar
                    json_match = re.search(r'\{[^{}]*"query"[^{}]*\}', generated_text)
                    if json_match:
                        query_json = orjson.loads(json_match.group())
                        prediction = query_json.get("query", "")
                    else:
                        # Fallback: use the whole generated text
                        prediction = generated_text
                except:
                    # If parsing fails, use the generated text as is
                    prediction = generated_text

                result = {"case_id": original_case["case_id"], "prediction": prediction}
                results.append(result)

            except Exception as e:
                print(f"Error processing case {original_case['case_id']}: {e}")
                # Add empty prediction on error
                results.append({"case_id": original_case["case_id"], "prediction": ""})

        print(f"{inference_mode.upper()} API inference complete.")

    # --- 4. Save Results in gold.json format ---
    print(f"Saving results to {output_file}...")

    with open(output_file, "w") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2).decode("utf-8"))

    print(f"Results saved to {output_file} in gold.json format")


if __name__ == "__main__":
    app()
