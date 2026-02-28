import json
from pathlib import Path
import xml.etree.ElementTree as ET


class ArchEHRSubtask4DataLoader:
    """Load and parse ArchEHR-QA data for Subtask 4: Evidence Alignment.

    Each case yields:
        - case_id: str
        - patient_question: str
        - clinician_question: str
        - note_sentences: str          (Formatted string for the prompt)
        - answer_sentences: str        (Formatted string for the prompt)
        - raw_answer_sentences: list   (Kept as dicts for inference fallback logic)
    """

    def __init__(self, xml_path: str | Path, json_path: str | Path):
        self.xml_path = Path(xml_path)
        self.json_path = Path(json_path)

    def load(self) -> list[dict]:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON not found: {self.json_path}")

        # 1. Parse the XML data for the context and note sentences
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        xml_data_map = {}
        for case in root.findall("case"):
            case_id = case.get("id")
            if case_id is None:
                raise ValueError("Found <case> without id")

            patient_question = case.findtext("patient_narrative", "").strip()
            clinician_question = case.findtext("clinician_question", "").strip()
            
            sentences_node = case.find("note_excerpt_sentences")
            if sentences_node is None:
                raise ValueError(f"Case {case_id}: missing note_excerpt_sentences")

            note_sentences = []
            seen_ids = set()

            for sent in sentences_node.findall("sentence"):
                sent_id_str = sent.get("id")
                if sent_id_str is None:
                    continue

                sent_id = int(sent_id_str)
                if sent_id in seen_ids:
                    continue

                text = (sent.text or "").strip()
                if text:
                    seen_ids.add(sent_id)
                    note_sentences.append({"sentence_id": str(sent_id), "text": text})

            xml_data_map[case_id] = {
                "patient_question": patient_question,
                "clinician_question": clinician_question,
                "note_sentences": note_sentences,
            }

        # 2. Parse the JSON data for the reference answer sentences
        with open(self.json_path, "r", encoding="utf-8") as f:
            json_data_list = json.load(f)

        cases = []
        
        # 3. Merge and format the data
        for item in json_data_list:
            case_id = item.get("case_id")
            if not case_id or case_id not in xml_data_map:
                continue
            
            xml_case = xml_data_map[case_id]
            
            raw_answer_sentences = []
            for ans_sent in item.get("clinician_answer_sentences", []):
                ans_id = ans_sent.get("id")
                ans_text = ans_sent.get("text", "").strip()
                if ans_id and ans_text:
                    raw_answer_sentences.append({
                        "answer_id": str(ans_id),
                        "text": ans_text
                    })

            # Format the lists into clean, readable strings for the LLM prompt
            formatted_note_sentences = "\n".join(
                [f"[{s['sentence_id']}] {s['text']}" for s in xml_case["note_sentences"]]
            )
            formatted_answer_sentences = "\n".join(
                [f"[{s['answer_id']}] {s['text']}" for s in raw_answer_sentences]
            )

            cases.append({
                "case_id": case_id,
                "patient_question": xml_case["patient_question"],
                "clinician_question": xml_case["clinician_question"],
                "note_sentences": formatted_note_sentences,
                "answer_sentences": formatted_answer_sentences,
                "raw_answer_sentences": raw_answer_sentences
            })

        return cases