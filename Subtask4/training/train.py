import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup

from inference.dataloader import ArchEHRSubtask4DataLoader
from inference.pairwise import build_pairwise_matrix
from training.dataset import PairwiseAlignmentDataset
from training.model import DebertaPairwiseClassifier


# ========================
# CONFIG
# ========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_GRAD_NORM = 1.0


# ========================
# DATA
# ========================

def build_dev_dataset():
    BASE_DIR = Path(__file__).resolve().parents[1]
    xml_path = BASE_DIR / "data" / "dev" / "archehr-qa.xml"
    key_path = BASE_DIR / "data" / "dev" / "archehr-qa_key.json"

    loader = ArchEHRSubtask4DataLoader(xml_path, key_path)
    return loader.load()


def split_cases(cases, train_ratio=0.75):
    random.shuffle(cases)  # important to avoid label ordering bias
    split_index = int(len(cases) * train_ratio)
    return cases[:split_index], cases[split_index:]


def build_pairs(cases):
    all_pairs = []

    for case in cases:
        pairs = build_pairwise_matrix(case)

        for p in pairs:
            if p["label"] is not None:
                all_pairs.append(
                    {
                        "answer_text": p["answer_text"],
                        "note_text": p["note_text"],
                        "label": float(p["label"]),  # force float early
                    }
                )

    return all_pairs


# ========================
# EVALUATION
# ========================

def evaluate(model, dataloader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE).float()

            logits = model(input_ids, attention_mask)

            # clamp logits for numerical safety
            logits = torch.clamp(logits, -50, 50)

            probs = torch.sigmoid(logits)

            preds.extend((probs > 0.5).int().cpu().tolist())
            labels.extend(y.int().cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return precision, recall, f1


# ========================
# TRAINING
# ========================

def train():

    cases = build_dev_dataset()
    train_cases, val_cases = split_cases(cases)

    train_pairs = build_pairs(train_cases)
    val_pairs = build_pairs(val_cases)

    train_dataset = PairwiseAlignmentDataset(train_pairs, MODEL_NAME)
    val_dataset = PairwiseAlignmentDataset(val_pairs, MODEL_NAME)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Using device: {DEVICE}")

    model = DebertaPairwiseClassifier(MODEL_NAME).to(DEVICE)
    model.float()

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # ===== CLASS WEIGHTING =====
    labels = [p["label"] for p in train_pairs]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)

    print(f"Positives: {num_pos} | Negatives: {num_neg}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).float()

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            logits = torch.clamp(logits, -50, 50)

            loss = loss_fn(logits, labels)

            if torch.isnan(loss):
                print("Loss exploded to NaN.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        precision, recall, f1 = evaluate(model, val_loader)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pt")
            print("✔ Saved new best model")

        print(f"\nEpoch {epoch+1}")
        print("Train Loss:", round(total_loss / len(train_loader), 6))
        print("Val Precision:", round(float(precision), 4))
        print("Val Recall:", round(float(recall), 4))
        print("Val F1:", round(float(f1), 4))


if __name__ == "__main__":
    train()