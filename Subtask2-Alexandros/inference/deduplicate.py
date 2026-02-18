"""
Section-aware deduplication for Subtask 2: Evidence Identification

After a model makes predictions, discharge / followup summary sections often
repeat what was already said in the hospital course section. This inflates FP
counts without adding recall. This script removes Zone B sentences that are
sufficiently similar to any Zone A sentence already in the prediction.

Zone A (authoritative): Brief Hospital Course, History of Present Illness,
                        Pertinent Results, and any other section not in Zone B.
Zone B (summaries):     Discharge Instructions, Followup Instructions,
                        Transitional Issues.

Similarity metric: Jaccard on word sets (case-insensitive, punctuation stripped).
A Zone B sentence is dropped when:  similarity(B, any_A) >= threshold

Usage (via deduplicate.sh):
    uv run python deduplicate.py \
        --input-file  ../outputs/dev/predictions.json \
        --output-file ../outputs/dev/predictions_dedup.json \
        --xml-file    ../../data/dev/archehr-qa.xml \
        --threshold   0.5
"""

import re
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Configuration: section names that are treated as summaries (Zone B)
# ---------------------------------------------------------------------------
ZONE_B_KEYWORDS = [
    "discharge instructions",
    "followup instructions",
    "follow-up instructions",
    "follow up instructions",
    "transitional issues",
    "discharge condition",
    "discharge disposition",
]


def is_zone_b(section_name: str) -> bool:
    norm = section_name.lower().strip().rstrip(":")
    return any(kw in norm for kw in ZONE_B_KEYWORDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common clinical-text stopwords to ignore during similarity comparison
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "was", "were", "is", "are", "be", "been", "has", "have",
    "had", "he", "she", "it", "his", "her", "its", "this", "that", "these",
    "those", "you", "your", "we", "our", "patient", "pt", "also", "not",
    "no", "as", "by", "from", "due", "any", "did", "do", "does", "about",
    "which", "who", "there", "their", "them", "they", "will", "would",
    "could", "should", "may", "please", "report",
}


def tokenize(text: str) -> set:
    """Lowercase content-word tokens, stopwords removed."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def containment(b: set, a_pool: set) -> float:
    """Fraction of B's content words that appear anywhere in A.
    Better than Jaccard when B is short and A is a large vocabulary pool.
    0.0 = no overlap, 1.0 = all of B's words are covered by A."""
    if not b:
        return 0.0
    return len(b & a_pool) / len(b)


def assign_sections(sentences: list[dict]) -> dict[str, str]:
    """
    Walk sentences in ID order; each sentence is assigned to the most recent
    section header seen before it. A sentence is treated as a section header
    if its text matches: all-caps-ish word(s) ending with a colon and ≤ 6 words.

    Returns: {sentence_id: section_name}
    """
    HEADER_RE = re.compile(r"^[A-Z][A-Za-z0-9 /\-]+:$")
    current_section = "unknown"
    assignment = {}

    for s in sorted(sentences, key=lambda x: int(x["sentence_id"])):
        text = s["text"].strip()
        word_count = len(text.split())
        if HEADER_RE.match(text) and word_count <= 6:
            current_section = text.rstrip(":")
            assignment[s["sentence_id"]] = current_section
        else:
            assignment[s["sentence_id"]] = current_section

    return assignment


def load_sentences_from_xml(xml_path: Path) -> dict[str, dict[str, str]]:
    """Returns {case_id: {sentence_id: sentence_text}}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    for case in root.findall("case"):
        case_id = case.get("id", "").strip()
        sents = {}
        for s in case.findall(".//note_excerpt_sentences/sentence"):
            sid = s.get("id", "").strip()
            text = (s.text or "").strip()
            sents[sid] = text
        result[case_id] = sents
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Section-aware deduplication: remove Zone-B sentences "
                    "that echo Zone-A sentences already in the prediction."
    )
    parser.add_argument("--input-file",  type=Path, required=True,
                        help="Prediction JSON to filter")
    parser.add_argument("--output-file", type=Path, required=True,
                        help="Where to write the deduplicated predictions")
    parser.add_argument("--xml-file",    type=Path, required=True,
                        help="archehr-qa.xml (for sentence texts)")
    parser.add_argument("--threshold",   type=float, default=0.3,
                        help="Jaccard similarity threshold on content words (default 0.3)")
    args = parser.parse_args()

    # --- Load ---
    with open(args.input_file) as f:
        submission = json.load(f)

    all_sentences = load_sentences_from_xml(args.xml_file)

    output = []
    total_removed = 0

    for case in submission:
        case_id   = case["case_id"]
        predicted = case["prediction"]   # list of sentence ID strings

        sent_map   = all_sentences.get(case_id, {})   # sid -> text
        all_sents  = [{"sentence_id": sid, "text": text}
                      for sid, text in sent_map.items()]
        sections   = assign_sections(all_sents)        # sid -> section name

        # Partition predicted IDs into Zone A and Zone B sets
        zone_a_ids = [sid for sid in predicted if not is_zone_b(sections.get(sid, ""))]
        zone_b_ids = [sid for sid in predicted if is_zone_b(sections.get(sid, ""))]

        # Pre-tokenise Zone A sentences and build a vocabulary pool.
        # Primary metric: containment — fraction of B's content words that
        # appear anywhere in Zone A. This catches paraphrases where clinical
        # jargon in Zone A maps to patient-friendly language in Zone B
        # (e.g. "TTE" ↔ "ultrasound of your heart").
        zone_a_tokens = [tokenize(sent_map.get(sid, "")) for sid in zone_a_ids]
        zone_a_pool   = set().union(*zone_a_tokens) if zone_a_tokens else set()

        kept_b = []
        removed_b = []
        for sid in zone_b_ids:
            b_tokens = tokenize(sent_map.get(sid, ""))
            # How much of B's vocabulary is already covered by Zone A?
            score = containment(b_tokens, zone_a_pool)
            if score >= args.threshold:
                removed_b.append(sid)
                print(
                    f"  Case {case_id}: removed Zone-B sentence {sid} "
                    f"(section='{sections.get(sid, '?')}') — "
                    f"containment={score:.0%} ≥ {args.threshold:.0%}"
                )
            else:
                kept_b.append(sid)

        final_prediction = sorted(
            zone_a_ids + kept_b,
            key=lambda x: int(x) if x.isdigit() else 0,
        )
        total_removed += len(removed_b)

        output.append({"case_id": case_id, "prediction": final_prediction})

    # --- Save ---
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] Deduplication complete.")
    print(f"  Sentences removed : {total_removed}")
    print(f"  Output            : {args.output_file}")


if __name__ == "__main__":
    main()
