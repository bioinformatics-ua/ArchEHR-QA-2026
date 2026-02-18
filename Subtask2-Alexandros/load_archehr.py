import json
import xml.etree.ElementTree as ET
from pathlib import Path


def load_all_cases(xml_path: str, key_path: str):
    """
    Load all ArchEHR cases with:
      - clinician question
      - sentence-level note excerpt sentences
    """

    xml_path = Path(xml_path)
    key_path = Path(key_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"Key JSON not found: {key_path}")

    # -----------------------
    # Load key JSON (filter cases)
    # -----------------------
    with open(key_path, "r") as f:
        key_json = json.load(f)

    valid_case_ids = {case["case_id"] for case in key_json}

    # -----------------------
    # Parse XML
    # -----------------------
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cases = []

    for case in root.findall("case"):
        case_id = case.attrib.get("id")

        if case_id not in valid_case_ids:
            continue

        # -----------------------
        # Clinician question
        # -----------------------
        cq_elem = case.find("clinician_question")
        if cq_elem is None or not cq_elem.text:
            continue

        clinician_question = cq_elem.text.strip()

        # -----------------------
        # Sentences
        # -----------------------
        sentences = {}

        sent_block = case.find("note_excerpt_sentences")
        if sent_block is None:
            continue

        for s in sent_block.findall("sentence"):
            sent_id = s.attrib.get("id")
            text = s.text.strip() if s.text else ""

            if sent_id and text:
                sentences[sent_id] = text

        if not sentences:
            continue

        cases.append({
            "case_id": case_id,
            "clinician_question": clinician_question,
            "sentences": sentences
        })

    return cases
