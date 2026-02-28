from typing import Dict, List


def compute_metrics(
    gold: Dict[str, List[str]],
    pred: Dict[str, List[str]],
):
    """
    Compute precision, recall, and F1 score
    for one case (all answer_ids).
    """

    tp = 0
    fp = 0
    fn = 0

    for answer_id, gold_notes in gold.items():
        pred_notes = pred.get(answer_id, [])

        gold_set = set(gold_notes)
        pred_set = set(pred_notes)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }