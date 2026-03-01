from pathlib import Path
import sys

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

    print("\n========================================")
    print("THRESHOLD SWEEP START")
    print("========================================")

    best_f1 = 0
    best_threshold = 0

    # Sweep thresholds from 0.05 to 0.95
    thresholds = [x / 100 for x in range(5, 100, 5)]

    for threshold in thresholds:

        cross_encoder = CrossEncoderClassifier(threshold=threshold)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        retriever_hits = 0
        retriever_total = 0

        for case in cases:

            gold = case["gold_alignments"]
            pred = {}

            for ans in case["answer_sentences"]:

                answer_id = ans["answer_id"]
                gold_notes = gold.get(answer_id, [])

                # -------------------------
                # RETRIEVER (Top-10)
                # -------------------------
                ranked = retriever.rank(
                    answer_text=ans["text"],
                    note_sentences=case["note_sentences"],
                    top_k=10,
                )

                top_ids = [r["note_id"] for r in ranked]

                # Retriever Recall@10 diagnostic
                for g in gold_notes:
                    retriever_total += 1
                    if g in top_ids:
                        retriever_hits += 1

                # -------------------------
                # CROSS ENCODER
                # -------------------------
                results = cross_encoder.classify(
                    answer_text=ans["text"],
                    candidate_notes=ranked,
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

        # -------------------------
        # GLOBAL METRICS
        # -------------------------
        precision = (
            total_tp / (total_tp + total_fp)
            if (total_tp + total_fp) > 0 else 0
        )

        recall = (
            total_tp / (total_tp + total_fn)
            if (total_tp + total_fn) > 0 else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        retriever_recall = (
            retriever_hits / retriever_total
            if retriever_total > 0 else 0
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"P {precision:.4f} | "
            f"R {recall:.4f} | "
            f"F1 {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\n========================================")
    print("BEST RESULT")
    print("========================================")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1: {best_f1:.4f}")
    print("========================================")


if __name__ == "__main__":
    run_dev()