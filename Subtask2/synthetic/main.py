"""
Generate synthetic sentence-level relevance labels for ArchEHR-QA test cases.

This script uses LLMs via the common providers package to classify each sentence
in the clinical notes as essential, supplementary, or not-relevant for answering
the clinician's question.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

import orjson
import typer
from common.dataloader import ArchEHRDataLoader, Case
from common.providers.base import BaseProvider
from common.providers.cloud import CloudProvider
from common.providers.local import LocalProvider

app = typer.Typer()


# System prompt that defines the task
SYSTEM_PROMPT = """You are a medical expert tasked with analyzing clinical notes to determine sentence relevance for answering patient questions.

For each sentence in the clinical note, you must classify it into one of three categories:
1. **essential**: The sentence contains critical information directly needed to answer the question
2. **supplementary**: The sentence provides supporting context or additional relevant details
3. **not-relevant**: The sentence does not contribute to answering the question

Be precise and objective in your assessment. Focus on what information is truly necessary versus merely contextual."""


def build_classification_prompt(case: Case) -> str:
    """Build a prompt for classifying all sentences in a case."""

    # Format sentences with their IDs
    sentences_text = "\n".join(f"[Sentence {s.id}]: {s.text}" for s in case.sentences)

    prompt = f"""
**Clinical Note Sentences:**
{sentences_text}

**Task:**
For each sentence above, determine its relevance for answering the clinical question. Classify each sentence as "essential", "supplementary", or "not-relevant".

Respond with a JSON object containing your classifications:
{{
    "classifications": [
        {{"sentence_id": "1", "relevance": "essential|supplementary|not-relevant"}},
        {{"sentence_id": "2", "relevance": "essential|supplementary|not-relevant"}},
        ...
    ]
}}

Provide ONLY the JSON object in your response, no additional text."""

    return prompt


def parse_llm_response(
    response: str, expected_sentence_ids: list[str]
) -> list[dict[str, str]]:
    """
    Parse the LLM response to extract sentence classifications.

    Args:
        response: Raw LLM response text
        expected_sentence_ids: List of sentence IDs that should be classified

    Returns:
        List of dicts with sentence_id and relevance keys
    """
    # Remove thinking from the response if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    response = response.strip("```")
    response = response.lstrip("json")

    try:
        # Try to parse as JSON
        parsed = orjson.loads(response)

        # Handle various response formats
        if "classifications" in parsed:
            classifications = parsed["classifications"]
        elif isinstance(parsed, list):
            classifications = parsed
        else:
            raise ValueError("Unexpected JSON structure")

        # Validate and normalize
        result = []
        for item in classifications:
            if "sentence_id" in item and "relevance" in item:
                relevance = item["relevance"].lower()
                # Normalize variations
                if relevance in ["essential", "supplementary", "not-relevant"]:
                    result.append(
                        {
                            "sentence_id": str(item["sentence_id"]),
                            "relevance": relevance,
                        }
                    )

        # Ensure all expected sentences are present
        found_ids = {item["sentence_id"] for item in result}
        missing_ids = set(expected_sentence_ids) - found_ids

        if missing_ids:
            print(f"Warning: Missing classifications for sentences: {missing_ids}")
            # Add default classifications for missing sentences
            for missing_id in missing_ids:
                result.append(
                    {"sentence_id": str(missing_id), "relevance": "not-relevant"}
                )

        # Sort by sentence_id
        result.sort(key=lambda x: int(x["sentence_id"]))

        return result

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response: {response[:500]}...")
        # Return default classifications
        return [
            {"sentence_id": sid, "relevance": "not-relevant"}
            for sid in expected_sentence_ids
        ]


@app.command()
def main(
    xml_file: Annotated[
        Path, typer.Option(help="Path to the XML file with cases and sentences.")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file for synthetic labels.")
    ],
    inference_mode: Annotated[
        Literal["local", "cloud"],
        typer.Option(
            help="Inference mode: 'local' for vLLM, 'cloud' for OpenRouter",
        ),
    ] = "cloud",
    model: Annotated[
        str, typer.Option(help="Model to use for generation.")
    ] = "anthropic/claude-3.5-sonnet",
) -> None:
    """
    Generate synthetic sentence-level relevance labels using LLMs.

    This script processes each case in the XML file, sending the patient narrative,
    clinician question, and all sentences to an LLM for classification.
    """

    # --- 1. Setup provider ---
    print(f"Setting up provider for inference mode: {inference_mode}...")
    provider: BaseProvider
    match inference_mode:
        case "local":
            provider = LocalProvider(model)
        case "cloud":
            provider = CloudProvider(model)
        case _:
            raise ValueError(f"Unknown inference mode: {inference_mode}")

    # --- 2. Load cases with sentences ---
    print(f"Loading cases from {xml_file}...")
    loader = ArchEHRDataLoader(xml_file)
    cases = loader.load()
    print(f"Loaded {len(cases)} cases from XML.")

    # --- 4. Process cases ---

    results: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        print(f"\nProcessing case {i + 1}/{len(cases)} (ID: {case.case_id})...")

        # Build prompts for batch
        user_prompt = build_classification_prompt(case)
        messages = provider.build_prompt(
            SYSTEM_PROMPT,
            f"""
**Patient Narrative:**
{case.patient_narrative}

**Clinical Question:**
{case.clinician_question}
""",
            user_prompt,
        )

        # Generate classifications
        response = provider.generate(messages)

        expected_ids = [s.id for s in case.sentences]
        classifications = parse_llm_response(response, expected_ids)

        results.append({"case_id": case.case_id, "answers": classifications})

    # --- 5. Final save ---
    print(f"\nAll cases processed. Saving final results to {output_file}...")
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

    print(f"✓ Successfully generated labels for {len(results)} cases.")
    print(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    app()
