"""
Per-sentence analysis script for Subtask 2: Evidence Identification

For each sentence across all cases, outputs:
  - case_id, sentence_id, relevance label, whether it was predicted,
    the sentence text, and whether it is TP/FP/FN/TN
  - Per-case precision, recall, F1 (strict: essential only)
  - Overall macro precision, recall, F1

Usage:
    python results_analysis.py \
        --submission_path outputs/dev/my_predictions.json \
        --key_path ../../data/dev/archehr-qa_key.json \
        --xml_path ../../data/dev/archehr-qa.xml \
        --out_file_path results-analysis/dev/my_predictions.json
"""

import json
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_submission(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {case["case_id"]: set(case["prediction"]) for case in data}


def load_key(path):
    """Returns dict: case_id -> {sentence_id -> relevance_label}"""
    with open(path, "r") as f:
        data = json.load(f)
    result = {}
    for case in data:
        result[case["case_id"]] = {
            a["sentence_id"]: a["relevance"] for a in case.get("answers", [])
        }
    return result


def load_sentence_texts(xml_path):
    """Returns dict: case_id -> {sentence_id -> sentence_text}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    for case in root.findall("case"):
        case_id = case.findtext("case_id", "").strip()
        sentences = {}
        for sent in case.findall(".//note_excerpt_sentences/sentence"):
            sid = sent.findtext("sentence_id", "").strip()
            text = sent.findtext("sentence_text", "").strip()
            sentences[sid] = text
        result[case_id] = sentences
    return result


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Per-sentence P/R/F1 analysis for Subtask 2")
    parser.add_argument("--submission_path", required=True, help="Path to prediction JSON")
    parser.add_argument("--key_path",        required=True, help="Path to key JSON")
    parser.add_argument("--xml_path",        required=True, help="Path to XML for sentence texts")
    parser.add_argument("--out_file_path",   required=True, help="Where to write analysis JSON")
    args = parser.parse_args()

    submissions  = load_submission(args.submission_path)
    key_map      = load_key(args.key_path)
    sent_texts   = load_sentence_texts(args.xml_path)

    cases_output = []
    all_case_precisions = []
    all_case_recalls    = []
    all_case_f1s        = []

    for case_id in sorted(key_map.keys(), key=lambda x: int(x)):
        relevances  = key_map[case_id]
        predicted   = submissions.get(case_id, set())
        texts       = sent_texts.get(case_id, {})

        # Gold = essential sentences only (strict)
        gold = {sid for sid, rel in relevances.items() if rel == "essential"}

        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)

        p, r, f1 = precision_recall_f1(tp, fp, fn)
        all_case_precisions.append(p)
        all_case_recalls.append(r)
        all_case_f1s.append(f1)

        # Build per-sentence rows
        sentences = []
        for sid in sorted(relevances.keys(), key=lambda x: int(x)):
            relevance   = relevances[sid]
            is_gold     = (relevance == "essential")
            is_pred     = (sid in predicted)

            if is_gold and is_pred:
                category = "TP"
            elif not is_gold and is_pred:
                category = "FP"
            elif is_gold and not is_pred:
                category = "FN"
            else:
                category = "TN"

            sentences.append({
                "sentence_id": sid,
                "text":        texts.get(sid, ""),
                "relevance":   relevance,
                "predicted":   is_pred,
                "category":    category,   # TP / FP / FN / TN (strict)
            })

        cases_output.append({
            "case_id":   case_id,
            "precision": p,
            "recall":    r,
            "f1":        f1,
            "tp":        tp,
            "fp":        fp,
            "fn":        fn,
            "sentences": sentences,
        })

    # Overall macro averages
    n = len(all_case_precisions)
    overall_precision = round(sum(all_case_precisions) / n, 4) if n else 0.0
    overall_recall    = round(sum(all_case_recalls)    / n, 4) if n else 0.0
    overall_f1        = round(sum(all_case_f1s)        / n, 4) if n else 0.0

    output = {
        "summary": {
            "overall_precision": overall_precision,
            "overall_recall":    overall_recall,
            "overall_f1":        overall_f1,
            "num_cases":         n,
        },
        "cases": cases_output,
    }

    # Write output
    out_path = Path(args.out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Overall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall F1        : {overall_f1:.4f}")
    print(f"Analysis written  : {out_path}")


if __name__ == "__main__":
    main()
