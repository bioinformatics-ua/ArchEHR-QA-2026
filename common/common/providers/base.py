import re
from abc import ABC, abstractmethod

import orjson
from openai.types.chat import ChatCompletionMessageParam

Messages = list[ChatCompletionMessageParam]


class BaseProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def build_prompt(
        self,
        system_prompt: str,
        case: str,
        user_prompt: str | None = None,
        fallback: bool = False,
    ) -> Messages:
        user_prompt = user_prompt or "Answer the question based on the above narrative."

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
                if not fallback
                else system_prompt,
            },  # pyright: ignore[reportReturnType] OpenAI doesn't support `cache_control` key
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": case,
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ]
                if not fallback
                else f"{case}\n\n{user_prompt}",
            },
        ]

    @abstractmethod
    def generate(self, prompt: Messages) -> str:
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[Messages]) -> list[str]:
        pass

    def parse_response(self, response: str) -> str:
        try:
            # Find JSON pattern {"query": "..."} or similar
            json_match = re.search(r'\{[^{}]*"query"[^{}]*\}', response)
            if not json_match:
                return response

            query_json = orjson.loads(json_match.group())
            return query_json.get("query", "")
        except Exception:
            return response
