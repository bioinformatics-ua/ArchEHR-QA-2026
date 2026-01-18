import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bert_score



# PATH HANDLING

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "archehr-qa-a-dataset-for-addressing-patients-information-needs-related-to-clinical-course-of-hospitalization-1.3",
    "dev"
)

KEY_PATH = os.path.join(DATA_DIR, "archehr-qa_key.json")
XML_PATH = os.path.join(DATA_DIR, "archehr-qa.xml")

# predictions file (your model outputs)
PREDICTIONS_PATH = os.path.join(DATA_DIR, "subtask1_predictions.json")



# HELPER

def truncate_to_15_words(text: str) -> str:
    return " ".join(text.split()[:15])



# DATA LOADER (SUBTASK 1)

class ArchEHRDataLoader:
    def __init__(self, key_path: str, xml_path: str):
        self.key_path = key_path
        self.xml_path = xml_path

    def load(self) -> List[Dict]:
        with open(self.key_path, "r", encoding="utf-8") as f:
            keys = {item["case_id"]: item for item in json.load(f)}

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        data = []
        for case in root.findall("case"):
            case_id = case.get("id")
            clinician_question = case.findtext("clinician_question", "").strip()

            if clinician_question:
                data.append({
                    "case_id": case_id,
                    "clinician_question": clinician_question
                })

        return data



# MAIN EVALUATION

def main():
    print(">>> Loading gold data")

    loader = ArchEHRDataLoader(KEY_PATH, XML_PATH)
    gold_cases = loader.load()

    print(f">>> Loaded {len(gold_cases)} gold cases")


    # Load predictions

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    pred_lookup = {
        item["case_id"]: item["prediction"]
        for item in preds
    }

    references = []
    predictions = []

    for case in gold_cases:
        case_id = case["case_id"]

        if case_id not in pred_lookup:
            continue

        gold_text = case["clinician_question"]
        pred_text = truncate_to_15_words(pred_lookup[case_id])

        references.append(gold_text)
        predictions.append(pred_text)

    print(f">>> Evaluating {len(predictions)} cases")

    if len(predictions) == 0:
        raise ValueError("No overlapping case_ids between gold and predictions.")


    # BLEU-5

    print(">>> BLEU START")
    bleu = BLEU(max_ngram_order=5)
    bleu_score = bleu.corpus_score(predictions, [references]).score
    print(f"BLEU-5: {bleu_score:.2f}")


    # ROUGE

    print(">>> ROUGE START")
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1 = r2 = rL = 0.0
    for pred, ref in zip(predictions, references):
        scores = rouge.score(ref, pred)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rL += scores["rougeL"].fmeasure

    n = len(predictions)
    r1 /= n
    r2 /= n
    rL /= n

    print(f"ROUGE-1 F1: {r1:.4f}")
    print(f"ROUGE-2 F1: {r2:.4f}")
    print(f"ROUGE-L F1: {rL:.4f}")

    # BERTScore

    print(">>> BERTSCORE START")
    _, _, F1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        device="cpu"
    )

    bert_f1 = F1.mean().item()
    print(f"BERTScore F1: {bert_f1:.4f}")

    print(">>> EVALUATION DONE")


if __name__ == "__main__":
    main()
