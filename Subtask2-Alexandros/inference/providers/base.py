from abc import ABC, abstractmethod
import re
from typing import Any

import orjson
from openai.types.chat import ChatCompletionMessageParam

Messages = list[ChatCompletionMessageParam]


class BaseProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def build_prompt(self, prompt_template: str, case: dict[str, Any]) -> Messages:
        """Substitute placeholders in the prompt template with case data."""
        content = prompt_template

        if "{PATIENT_QUESTION}" in content:
            content = content.replace(
                "{PATIENT_QUESTION}", str(case.get("patient_question", ""))
            )
        if "{CLINICIAN_QUESTION}" in content:
            content = content.replace(
                "{CLINICIAN_QUESTION}", str(case.get("clinician_question", ""))
            )
        if "{SENTENCES}" in content:
            content = content.replace(
                "{SENTENCES}", str(case.get("sentences", ""))
            )
        if "{SENTENCE}" in content:
            content = content.replace(
                "{SENTENCE}", str(case.get("sentence", ""))
            )
        if "{CASE_ID}" in content:
            content = content.replace(
                "{CASE_ID}", str(case.get("case_id", ""))
            )

        return [{"role": "user", "content": content}]

    @abstractmethod
    def generate(self, prompt: Messages) -> str:
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[Messages]) -> list[str]:
        pass

    def parse_response(self, response: str) -> dict | None:
        """Extract first JSON object containing 'prediction' from model output.

        Returns the parsed dict with keys 'case_id' and 'prediction',
        or None if no valid JSON is found.
        """
        # 1. Strip think-blocks (Qwen3 / deepseek style)
        cleaned = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL
        ).strip()

        # 2. Remove markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 3. Try to find a JSON object
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = orjson.loads(json_match.group())
                if "prediction" in data:
                    return data
            except Exception:
                pass

        return None
