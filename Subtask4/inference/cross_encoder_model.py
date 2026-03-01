from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch


class CrossEncoderClassifier:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.0,
    ):
        self.model = CrossEncoder(model_name)
        self.threshold = threshold

    @torch.no_grad()
    def classify(
        self,
        answer_text: str,
        candidate_notes: List[Dict],
    ) -> List[Dict]:

        pairs = [
            (answer_text, note["note_text"])
            for note in candidate_notes
        ]

        scores = self.model.predict(pairs)

        results = []

        for note, score in zip(candidate_notes, scores):
            support = score >= self.threshold

            results.append(
                {
                    "note_id": note["note_id"],
                    "score": float(score),
                    "support": support,
                }
            )

        return results