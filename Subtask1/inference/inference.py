from pathlib import Path
from typing import Annotated, Literal

import orjson
import typer
from common.dataloader import ArchEHRDataLoader, Case
from common.providers.base import BaseProvider, Messages
from common.providers.cloud import CloudProvider
from common.providers.local import LocalProvider

app = typer.Typer()


@app.command()
def main(
    xml_file: Annotated[
        Path, typer.Option(help="Path to the XML file with patient narratives.")
    ],
    prompt_file: Annotated[
        Path, typer.Option(help="Path to the prompt template JSONL file.")
    ],
    prompt_index: Annotated[int, typer.Option(help="Prompt id.")],
    output_file: Annotated[
        Path, typer.Option(help="Path to the output JSON file for results.")
    ],
    inference_mode: Annotated[
        Literal["local", "cloud"],
        typer.Option(
            help="Inference mode: 'local' for vLLM, 'cloud' for OpenRouter",
        ),
    ] = "local",
    model: Annotated[
        str, typer.Option(help="Hugging Face model to use.")
    ] = "Qwen/Qwen3-8B",
) -> None:
    # --- 1. Setup based on inference mode ---
    print(f"Setting up provider for inference mode: {inference_mode}...")
    match inference_mode:
        case "local":
            provider: BaseProvider = LocalProvider(model)
        case "cloud":
            provider: BaseProvider = CloudProvider(model)
        case _:
            raise ValueError(f"Unknown inference mode: {inference_mode}")

    # --- 2. Load XML Cases ---
    print(f"Loading XML cases from {xml_file}...")
    xml_cases = ArchEHRDataLoader(xml_file).load()
    print(f"Loaded {len(xml_cases)} cases from XML.")

    # --- 3. Load Prompt Template ---
    print(f"Loading prompt template from {prompt_file}...")
    with open(prompt_file, "r") as f:
        prompt_dict = orjson.loads(f.read())
    prompt_template = prompt_dict[str(prompt_index)]

    # list[(case, prompt)]
    p: list[tuple[Case, Messages]] = [
        (case, provider.build_prompt(prompt_template, case.clinician_question))
        for case in xml_cases
    ]
    print(f"Built {len(p)} prompts.")

    outputs = provider.batch_generate([prompt for _, prompt in p])

    results = [
        {
            "case_id": case.case_id,
            "prediction": provider.parse_response(output),
        }
        for case, output in zip([case for case, _ in p], outputs)
    ]

    print(f"Saving results to {output_file}...")
    with open(output_file, "w") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2).decode("utf-8"))

    print(f"Results saved to {output_file}. Finished.")


if __name__ == "__main__":
    app()
