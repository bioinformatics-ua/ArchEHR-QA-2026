import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from load_archehr import load_all_cases
# -----------------------
# CONFIG
# -----------------------

MODEL_NAME = "sharraks97/biorecall-PubMedBERT-NLI"
TOKENIZER_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENTAILMENT_THRESHOLD = 0.1  

XML_PATH = "/ceph/home/student.aau.dk/lj02sb/ArchEHR/atificial_data/archehr-qa/dev/archehr-qa.xml"
KEY_PATH = "/ceph/home/student.aau.dk/lj02sb/ArchEHR/atificial_data/archehr-qa/dev/archehr-qa_key.json"
OUT_PATH = "/ceph/home/student.aau.dk/lj02sb/ArchEHR/atificial_data/NLI_APPROACH/submission.json"

# -----------------------
# LOAD MODEL
# -----------------------

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

label2id = model.config.label2id
ENTAIL_ID = label2id["entailment"]

# -----------------------
# NLI SCORING FUNCTION
# -----------------------

@torch.no_grad()
def entailment_score(premise: str, hypothesis: str) -> float:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]
    return probs[ENTAIL_ID].item()

# -----------------------
# MAIN
# -----------------------

def main():
    print("Loading cases...")
    cases = load_all_cases(XML_PATH, KEY_PATH)
    print(f"Loaded {len(cases)} cases")

    submission = []

    for case in cases:
        case_id = case["case_id"]
        question = case["clinician_question"]
        sentences = case["sentences"]  # dict: id -> text

        predicted_ids = []

        for sent_id, sent_text in sentences.items():
            score = entailment_score(sent_text, question)
            if score >= ENTAILMENT_THRESHOLD:
                predicted_ids.append(sent_id)

        submission.append({
            "case_id": case_id,
            "prediction": predicted_ids
        })

    print(f"Saving submission to {OUT_PATH}")
    with open(OUT_PATH, "w") as f:
        json.dump(submission, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
