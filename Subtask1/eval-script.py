import os
import json
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Set

import torch
import nltk

# Metrics Imports
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import spacy
from alignscore import AlignScore

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# PATH HANDLING
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dev"
    # "test"
)

KEY_PATH = os.path.join(DATA_DIR, "archehr-qa_key.json")
XML_PATH = os.path.join(DATA_DIR, "archehr-qa.xml")

# CONFIGURATION FOR NEW METRICS
ALIGNSCORE_CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "AlignScore-base.ckpt")
MEDCON_SPACY_MODEL = "en_core_sci_sm"

# DEVICE HANDLING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Global Device Setting: {DEVICE.upper()}")


# HELPER FUNCTIONS
def truncate_to_15_words(text: str) -> str:
    return " ".join(text.split()[:15])


# METRIC CLASSES
class MedConScorer:
    """
    Implements Medical Concept Match (MEDCON) using scispaCy.
    """
    def __init__(self, model_name: str = "en_core_sci_sm"):
        print(f">>> Loading Spacy model for MEDCON: {model_name}")
        
        if DEVICE == "cuda":
            is_gpu = spacy.prefer_gpu()
            print(f">>> Spacy GPU enabled: {is_gpu}")
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"Could not load spacy model '{model_name}'. "
                f"Please install it via: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{model_name}-0.5.1.tar.gz"
            )

    def extract_concepts(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        return {ent.text.lower() for ent in doc.ents}

    def score(self, references: List[str], predictions: List[str]) -> float:
        total_f1 = 0.0
        n = len(references)

        for ref, pred in zip(references, predictions):
            ref_concepts = self.extract_concepts(ref)
            pred_concepts = self.extract_concepts(pred)

            if not ref_concepts and not pred_concepts:
                f1 = 1.0
            elif not ref_concepts or not pred_concepts:
                f1 = 0.0
            else:
                intersection = len(ref_concepts.intersection(pred_concepts))
                precision = intersection / len(pred_concepts)
                recall = intersection / len(ref_concepts)
                
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
            
            total_f1 += f1

        return total_f1 / n if n > 0 else 0.0


# DATA LOADER
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
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Evaluate predictions and output metrics to JSON.")
    parser.add_argument("prediction_file", type=str, help="Path to the prediction JSON file")
    args = parser.parse_args()

    prediction_path = args.prediction_file
    
    # 2. Load Gold Data
    print(">>> Loading gold data")
    loader = ArchEHRDataLoader(KEY_PATH, XML_PATH)
    gold_cases = loader.load()
    print(f">>> Loaded {len(gold_cases)} gold cases")

    # 3. Load Predictions
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Predictions file not found: {prediction_path}")

    print(f">>> Loading predictions from: {prediction_path}")
    with open(prediction_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    pred_lookup = {item["case_id"]: item["prediction"] for item in preds}

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

    # Dictionary to hold all metrics
    metrics_data = {}

    # 4. BLEU-5
    print("\n>>> BLEU START")
    bleu = BLEU(max_ngram_order=5)
    bleu_score = bleu.corpus_score(predictions, [references]).score
    metrics_data["BLEU-5"] = bleu_score
    print(f"BLEU-5: {bleu_score:.2f}")

    # 5. ROUGE
    print("\n>>> ROUGE START")
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        scores = rouge.score(ref, pred)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rL += scores["rougeL"].fmeasure
    
    n = len(predictions)
    metrics_data["ROUGE-1"] = r1/n
    metrics_data["ROUGE-2"] = r2/n
    metrics_data["ROUGE-L"] = rL/n
    print(f"ROUGE-1 F1: {r1/n:.4f}")

    # 6. BERTScore
    print(f"\n>>> BERTSCORE START (Device: {DEVICE})")
    _, _, F1 = bert_score(
        predictions, 
        references, 
        lang="en", 
        model_type="distilbert-base-uncased", 
        device=DEVICE
    )
    bert_mean = F1.mean().item()
    metrics_data["BERTScore"] = bert_mean
    print(f"BERTScore F1: {bert_mean:.4f}")

    # 7. AlignScore
    print(f"\n>>> ALIGNSCORE START (Device: {DEVICE})")
    metrics_data["AlignScore"] = None
    if os.path.exists(ALIGNSCORE_CKPT_PATH):
        try:
            scorer = AlignScore(
                model='roberta-base', 
                batch_size=32, 
                device=DEVICE,
                ckpt_path=ALIGNSCORE_CKPT_PATH, 
                evaluation_mode='nli_sp'
            )
            align_scores = scorer.score(contexts=references, claims=predictions)
            avg_align = sum(align_scores) / len(align_scores)
            metrics_data["AlignScore"] = avg_align
            print(f"AlignScore: {avg_align:.4f}")
        except Exception as e:
            print(f"AlignScore Error: {e}")
            metrics_data["AlignScore_Error"] = str(e)
    else:
        print(f"WARNING: AlignScore checkpoint not found at {ALIGNSCORE_CKPT_PATH}. Skipping.")
        metrics_data["AlignScore_Error"] = "Checkpoint not found"

    # 8. MEDCON
    print("\n>>> MEDCON START")
    metrics_data["MEDCON"] = None
    try:
        med_scorer = MedConScorer(model_name=MEDCON_SPACY_MODEL)
        med_f1 = med_scorer.score(references, predictions)
        metrics_data["MEDCON"] = med_f1
        print(f"MEDCON F1: {med_f1:.4f}")
    except Exception as e:
        print(f"MEDCON Failed: {e}")
        metrics_data["MEDCON_Error"] = str(e)

# 9. Write Output
    output_dir = "results"

    base_name = os.path.basename(prediction_path)
    output_path = os.path.join(output_dir, base_name)
    
    print(f"\n>>> Saving metrics to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print(">>> EVALUATION DONE")

if __name__ == "__main__":
    main()