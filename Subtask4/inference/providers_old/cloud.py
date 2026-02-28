import os
from typing import List
from openai import OpenAI

from .base import BaseProvider, Messages


class CloudProvider(BaseProvider):
    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.3,
        top_p: float = 0.90,
        max_tokens: int = 512,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        super().__init__(model_name)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # OpenRouter key
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY env variable is required")

        # OpenAI SDK + OpenRouter headers (required for POST)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://archehr-qa",
                "X-Title": "ArchEHR-QA-Subtask4",
                "Accept": "application/json",
            },
        )

    def generate(self, prompt: Messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""

    def batch_generate(self, prompts: List[Messages]) -> List[str]:
        return [self.generate(prompt) for prompt in prompts]
