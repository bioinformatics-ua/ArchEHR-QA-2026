from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict

import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")


class SBertRetriever:
    """
    SBERT bi-encoder retrieval stage.
    For each answer sentence, ranks note sentences by cosine similarity.
    """

    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    @torch.no_grad()
    def rank(
        self,
        answer_text: str,
        note_sentences: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Returns top_k note sentences ranked by similarity.
        """

        note_texts = [n["text"] for n in note_sentences]

        # Encode
        answer_emb = self.model.encode(
            answer_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        note_embs = self.model.encode(
            note_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Cosine similarity (since normalized → dot = cosine)
        scores = torch.matmul(note_embs, answer_emb)

        # Get top-k
        top_k = min(top_k, len(note_sentences))
        values, indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(values, indices):
            note = note_sentences[idx]
            results.append(
                {
                    "note_id": note["sentence_id"],
                    "note_text": note["text"],
                    "score": float(score.item()),
                }
            )

        return results