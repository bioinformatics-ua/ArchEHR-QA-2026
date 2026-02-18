from abc import ABC, abstractmethod
import re
from typing import Any, List, Dict

from openai.types.chat import ChatCompletionMessageParam
import orjson

# Python 3.8 compatible type alias
Messages = List[ChatCompletionMessageParam]


class BaseProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def build_prompt(self, prompt_template: str, case: Dict[str, Any]) -> Messages:
        content = prompt_template.replace("{CLINICIAN_QUESTION}", str(case.get("clinician_question", "")))
        
        if "{PATIENT_QUESTION}" in content:
            content = content.replace("{PATIENT_QUESTION}", str(case.get("patient_question", "")))
        if "{SENTENCES}" in content:
            content = content.replace("{SENTENCES}", str(case.get("sentences", "")))
        if "{SENTENCE}" in content:
            content = content.replace("{SENTENCE}", str(case.get("sentence", "")))
        if "{CASE_ID}" in content:
            content = content.replace("{CASE_ID}", str(case.get("case_id", "")))
            
        return [
            {
                "role": "user",
                "content": content,
            }
        ]


    @abstractmethod
    def generate(self, prompt: Messages) -> str:
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[Messages]) -> List[str]:
        pass

    def parse_response(self, response: str) -> str:
        try:
            json_match = re.search(r"\{[^{}]*\}", response)
            if not json_match:
                return response

            data = orjson.loads(json_match.group())
            return data.get("label", response)
        except Exception:
            return response
