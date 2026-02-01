import os
from typing import Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from .base import BaseProvider, Messages


class LocalProvider(BaseProvider):
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        super().__init__(model_name)
        hf_token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=2048,
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=1024
        )

    def build_prompt(
        self, prompt_template: str, case: dict[str, Any]
    ) -> Messages:
        return self.tokenizer.apply_chat_template(
            super().build_prompt(prompt_template, case),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def generate(self, prompt: Messages) -> str:
        return (
            self.llm.generate(prompt, self.sampling_params)[0].outputs[0].text.strip()  # pyright: ignore[reportArgumentType]
        )

    def batch_generate(self, prompts: list[Messages]) -> list[str]:
        return [
            output.outputs[0].text.strip()
            for output in self.llm.generate(prompts, self.sampling_params)  # pyright: ignore[reportArgumentType]
        ]
