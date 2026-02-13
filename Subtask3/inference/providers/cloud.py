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

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY env variable is required")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://archehr-qa",
                "X-Title": "ArchEHR-QA-Subtask3",
                "Accept": "application/json",
            },
        )

    def generate(self, prompt: Messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def batch_generate(self, prompts: list[Messages]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]
