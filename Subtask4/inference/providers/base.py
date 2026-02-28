from abc import ABC, abstractmethod
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

Messages = list[ChatCompletionMessageParam]


class BaseProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def build_prompt(self, prompt_template: str, case: dict[str, Any]) -> Messages:
        """Substitute placeholders in the prompt template with Subtask 4 case data."""
        content = prompt_template

        # --- Subtask 4 Placeholders ---
        if "{case_id}" in content:
            content = content.replace(
                "{case_id}", str(case.get("case_id", ""))
            )
        if "{patient_question}" in content:
            content = content.replace(
                "{patient_question}", str(case.get("patient_question", ""))
            )
        if "{clinician_question}" in content:
            content = content.replace(
                "{clinician_question}", str(case.get("clinician_question", ""))
            )
        if "{note_sentences}" in content:
            content = content.replace(
                "{note_sentences}", str(case.get("note_sentences", ""))
            )
        if "{answer_sentences}" in content:
            content = content.replace(
                "{answer_sentences}", str(case.get("answer_sentences", ""))
            )

        return [{"role": "user", "content": content}]

    @abstractmethod
    def generate(self, prompt: Messages) -> str:
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[Messages]) -> list[str]:
        pass