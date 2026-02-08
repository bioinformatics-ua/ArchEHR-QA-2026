import os
from openai import OpenAI

from .base import BaseProvider, Messages


class CloudProvider(BaseProvider):
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        super().__init__(model_name)
        if not (api_key := os.environ.get("OPENAI_API_KEY")):
            raise ValueError("OpenAI API key required via OPENAI_API_KEY env variable")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: Messages) -> str:
        print("Generating response from CloudProvider...")
        content = (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=0.7,
                max_tokens=8192,
            )
            .choices[0]
            .message.content
        )
        return content.strip() if content else ""

    def batch_generate(self, prompts: list[Messages]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]
