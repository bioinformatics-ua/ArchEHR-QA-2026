from abc import ABC, abstractmethod
import re
from typing import Any

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
        if "{NOTE_EXCERPT}" in content:
            content = content.replace(
                "{NOTE_EXCERPT}", str(case.get("note_excerpt", ""))
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

    def parse_response(self, response: str) -> str:
        """Extract the answer string from the model response.

        Tries JSON extraction first ({"answer": "..."}), then falls back
        to returning the raw response stripped of common wrapper artefacts.
        """
        # 1. Try JSON with an "answer" key
        try:
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response)
            if json_match:
                import orjson

                data = orjson.loads(json_match.group())
                answer = data.get("answer", "").strip()
                if answer:
                    return answer
        except Exception:
            pass

        # 2. Strip think-blocks (Qwen3 / deepseek style)
        cleaned = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL
        ).strip()

        # 3. Remove markdown code fences if the model wrapped its answer
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 4. Strip citations: [1], [**5**, **6**], (sentence 9), (sentences 2, 4), (2, 7), etc.
        cleaned = re.sub(r"\[[\d\s,\*\*]+\]", "", cleaned)          # [1], [**5**, **6**]
        cleaned = re.sub(r"\(sentences?\s*[\d\s,]+\)", "", cleaned)  # (sentence 9), (sentences 2, 4, 5)
        cleaned = re.sub(r"\(\d[\d\s,]*\)", "", cleaned)             # (2), (6, 7)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)                    # collapse double spaces

        return cleaned.strip()
