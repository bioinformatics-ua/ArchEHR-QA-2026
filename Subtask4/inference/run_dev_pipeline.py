from pathlib import Path

import sys
from pathlib import Path

# Add parent directory (Subtask4) to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference.dataloader import ArchEHRSubtask4DataLoader
from inference.retriever import SBertRetriever
from inference.cross_encoder_model import CrossEncoderClassifier
from evaluation.evaluator import compute_metrics


def run_dev():

    BASE_DIR = Path(__file__).resolve().parents[1]

    xml_path = BASE_DIR / "data" / "dev" / "archehr-qa.xml"
    key_path = BASE_DIR / "data" / "dev" / "archehr-qa_key.json"

    loader = ArchEHRSubtask4DataLoader(xml_path, key_path)
    cases = loader.load()

    retriever = SBertRetriever()
    cross_encoder = CrossEncoderClassifier(threshold=0.5)

    total_tp = total_fp = total_fn = 0

    for case in cases:

        gold = case["gold_alignments"]
        pred = {}

        for ans in case["answer_sentences"]:

            answer_id = ans["answer_id"]

            ranked = retriever.rank(
                answer_text=ans["text"],
                note_sentences=case["note_sentences"],
                top_k=len(case["note_sentences"]),
            )

            # dynamic filtering
            top_score = ranked[0]["score"]
            margin = 0.25

            filtered = [
                r for r in ranked
                if r["score"] >= top_score - margin
            ]

            results = cross_encoder.classify(
                answer_text=ans["text"],
                candidate_notes=filtered,
            )

            supported_notes = [
                r["note_id"]
                for r in results
                if r["support"]
            ]

            pred[answer_id] = supported_notes

        metrics = compute_metrics(gold, pred)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)

    print("\nDEV RESULTS")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))


if __name__ == "__main__":
    run_dev()