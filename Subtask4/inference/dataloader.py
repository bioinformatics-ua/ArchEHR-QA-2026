from pathlib import Path
import json
import xml.etree.ElementTree as ET
from typing import Optional


class ArchEHRSubtask4DataLoader:
    """
    Load and parse ArchEHR-QA data for Subtask 4: Evidence Alignment (Mode B).

    Produces per case:

    {
        "case_id": str,
        "answer_sentences": [
            {"answer_id": str, "text": str}
        ],
        "note_sentences": [
            {"sentence_id": str, "text": str}
        ],
        "gold_alignments": {           # only if key_json provided (dev)
            answer_id: [note_ids]
        }
    }
    """

    def __init__(
        self,
        xml_path: Path,
        key_json_path: Optional[Path] = None,
    ):
        self.xml_path = xml_path
        self.key_json_path = key_json_path

    def load(self) -> list[dict]:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        # --- Load XML ---
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # --- Load key JSON (dev only) ---
        key_data = {}
        if self.key_json_path:
            if not self.key_json_path.exists():
                raise FileNotFoundError(f"Key JSON not found: {self.key_json_path}")

            with open(self.key_json_path, "r", encoding="utf-8") as f:
                raw_key = json.load(f)

            # index by case_id
            key_data = {entry["case_id"]: entry for entry in raw_key}

        cases: list[dict] = []

        for case in root.findall("case"):
            case_id = case.get("id")
            if case_id is None:
                raise ValueError("Found <case> without id")

            # -------------------------
            # NOTE SENTENCES (XML)
            # -------------------------
            sentences_node = case.find("note_excerpt_sentences")
            if sentences_node is None:
                raise ValueError(f"Case {case_id}: missing note_excerpt_sentences")

            note_sentences = []
            seen_ids = set()

            for sent in sentences_node.findall("sentence"):
                sent_id = sent.get("id")
                if sent_id is None:
                    raise ValueError(f"Case {case_id}: sentence without id")

                if sent_id in seen_ids:
                    raise ValueError(
                        f"Case {case_id}: duplicate sentence id {sent_id}"
                    )

                text = (sent.text or "").strip()
                if not text:
                    raise ValueError(
                        f"Case {case_id}: empty text in sentence {sent_id}"
                    )

                seen_ids.add(sent_id)

                note_sentences.append(
                    {
                        "sentence_id": sent_id,
                        "text": text,
                    }
                )

            # -------------------------
            # ANSWER SENTENCES (JSON)
            # -------------------------
            answer_sentences = []
            gold_alignments = {}

            if case_id in key_data:
                key_entry = key_data[case_id]

                for ans in key_entry["clinician_answer_sentences"]:
                    answer_id = ans["id"]
                    answer_text = ans["text"].strip()

                    answer_sentences.append(
                        {
                            "answer_id": answer_id,
                            "text": answer_text,
                        }
                    )

                    # gold citations (dev only)
                    citations_raw = ans.get("citations")

                    if citations_raw:
                        # citations may be string like "2" or "2,5"
                        note_ids = [
                            c.strip()
                            for c in citations_raw.split(",")
                            if c.strip()
                        ]
                        gold_alignments[answer_id] = note_ids
                    else:
                        gold_alignments[answer_id] = []

            else:
                # test split → no gold
                answer_sentences = []
                gold_alignments = None

            cases.append(
                {
                    "case_id": case_id,
                    "answer_sentences": answer_sentences,
                    "note_sentences": note_sentences,
                    "gold_alignments": gold_alignments,
                }
            )

        return cases