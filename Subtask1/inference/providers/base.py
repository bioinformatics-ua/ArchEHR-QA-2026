from abc import ABC, abstractmethod
import re
from typing import Any
from openai.types.chat import ChatCompletionMessageParam
import orjson

Messages = list[ChatCompletionMessageParam]


class BaseProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def build_prompt(self, prompt_template: str, case: dict[str, Any]) -> Messages:
        return [
            {
                "role": "user",
                "content": prompt_template.replace(
                    "{PATIENT_NARRATIVE}", case["clinician_question"]
                ),
            }
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
