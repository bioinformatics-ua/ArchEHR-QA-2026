def build_pairwise_matrix(case: dict) -> list[dict]:
    """
    Build pairwise (answer_sentence, note_sentence) matrix.

    Returns list of dict:
    {
        case_id,
        answer_id,
        note_id,
        answer_text,
        note_text,
        label  # 1 if gold support else 0 (None if no gold)
    }
    """

    case_id = case["case_id"]
    answer_sentences = case["answer_sentences"]
    note_sentences = case["note_sentences"]
    gold_alignments = case["gold_alignments"]

    pairs = []

    for ans in answer_sentences:
        answer_id = ans["answer_id"]
        answer_text = ans["text"]

        gold_notes = []
        if gold_alignments is not None:
            gold_notes = gold_alignments.get(answer_id, [])

        for note in note_sentences:
            note_id = note["sentence_id"]
            note_text = note["text"]

            if gold_alignments is not None:
                label = 1 if note_id in gold_notes else 0
            else:
                label = None

            pairs.append(
                {
                    "case_id": case_id,
                    "answer_id": answer_id,
                    "note_id": note_id,
                    "answer_text": answer_text,
                    "note_text": note_text,
                    "label": label,
                }
            )

    return pairs