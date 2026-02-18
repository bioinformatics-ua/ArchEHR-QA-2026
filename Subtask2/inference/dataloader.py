from pathlib import Path
import xml.etree.ElementTree as ET


class ArchEHRSubtask2DataLoader:
    def __init__(self, xml_path: Path):
        self.xml_path = xml_path

    def load(self):
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        cases = []

        for case in root.findall("case"):
            case_id = case.get("id")
            if case_id is None:
                raise ValueError("Found <case> without id")

            clinician_question = (
                case.findtext("clinician_question", "").strip()
            )
            if not clinician_question:
                raise ValueError(f"Case {case_id}: missing clinician_question")

            patient_question = (
                case.findtext("patient_narrative", "").strip()
            )

            sentences_node = case.find("note_excerpt_sentences")
            if sentences_node is None:
                raise ValueError(f"Case {case_id}: missing note_excerpt_sentences")

            sentences = []
            seen_ids = set()

            for sent in sentences_node.findall("sentence"):
                sent_id = sent.get("id")
                if sent_id is None:
                    raise ValueError(f"Case {case_id}: sentence without id")

                sent_id = int(sent_id)
                if sent_id in seen_ids:
                    raise ValueError(f"Case {case_id}: duplicate sentence id {sent_id}")

                text = (sent.text or "").strip()
                if not text:
                    raise ValueError(f"Case {case_id}: empty text in sentence {sent_id}")

                seen_ids.add(sent_id)
                sentences.append(
                    {
                        "sentence_id": sent_id,
                        "text": text,
                    }
                )

            cases.append(
                {
                    "case_id": case_id,
                    "clinician_question": clinician_question,
                    "patient_question": patient_question,
                    "sentences": sentences,
                }
            )

        return cases
