from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class PairwiseAlignmentDataset(Dataset):
    """
    Dataset for pairwise (answer_sentence, note_sentence) classification.
    """

    def __init__(self, pairs, model_name, max_length=256):

        self.pairs = pairs
        self.max_length = max_length

        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        item = self.pairs[idx]

        # Safety: ensure strings are valid
        answer_text = str(item["answer_text"]) if item["answer_text"] is not None else ""
        note_text = str(item["note_text"]) if item["note_text"] is not None else ""

        # Avoid completely empty pairs
        if answer_text.strip() == "":
            answer_text = "[EMPTY]"
        if note_text.strip() == "":
            note_text = "[EMPTY]"

        encoding = self.tokenizer(
            answer_text,
            note_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Force label to strict float32 0.0 or 1.0
        label = float(item["label"])
        label = 1.0 if label == 1 else 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.float32),
        }