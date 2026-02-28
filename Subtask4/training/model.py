from transformers import AutoModel
import torch
import torch.nn as nn


class DebertaPairwiseClassifier(nn.Module):
    """
    Binary classifier for answer-note alignment.
    """

    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0]
        cls_embedding = cls_embedding.float()  # force float32
        logits = self.classifier(cls_embedding)

        return logits.squeeze(-1)