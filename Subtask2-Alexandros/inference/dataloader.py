from pathlib import Path
import xml.etree.ElementTree as ET


class ArchEHRSubtask2DataLoader:
    """Load and parse ArchEHR-QA XML data for Subtask 2: Evidence Identification.

    Each case yields:
        - case_id: str
        - patient_question: str   (the full patient narrative)
        - clinician_question: str (the concise clinician-interpreted question)
        - sentences: list[dict]   (id + text per numbered sentence)
    """

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path

    def load(self) -> list[dict]:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        cases: list[dict] = []

        for case in root.findall("case"):
            case_id = case.get("id")
            if case_id is None:
                raise ValueError("Found <case> without id")

            # --- Clinical specialty ---
            clinical_specialty = case.findtext("clinical_specialty", "").strip()

            # --- Clinician question ---
            clinician_question = case.findtext("clinician_question", "").strip()
            if not clinician_question:
                raise ValueError(f"Case {case_id}: missing clinician_question")

            # --- Patient narrative ---
            patient_question = case.findtext("patient_narrative", "").strip()

            # --- Numbered note sentences ---
            sentences_node = case.find("note_excerpt_sentences")
            if sentences_node is None:
                raise ValueError(
                    f"Case {case_id}: missing note_excerpt_sentences"
                )

            sentences: list[dict] = []
            seen_ids: set[int] = set()

            for sent in sentences_node.findall("sentence"):
                sent_id_str = sent.get("id")
                if sent_id_str is None:
                    raise ValueError(f"Case {case_id}: sentence without id")

                sent_id = int(sent_id_str)
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
                sentences.append({"sentence_id": sent_id, "text": text})

            cases.append(
                {
                    "case_id": case_id,
                    "clinical_specialty": clinical_specialty,
                    "clinician_question": clinician_question,
                    "patient_question": patient_question,
                    "sentences": sentences,
                }
            )

        return cases
