import json
import xml.etree.ElementTree as ET


def load_all_cases(xml_path: str, key_path: str):
    """
    Load ArchEHR cases using OFFICIAL sentence splits and labels.

    Returns a list of dicts:
    {
        case_id: str
        clinician_question: str
        sentences: { sentence_id (str) : sentence_text (str) }
        gold_labels: { sentence_id (str) : relevance (str) }   # optional, for eval/debug
    }
    """

    # -----------------------
    # Load key (labels)
    # -----------------------

    with open(key_path, "r") as f:
        key_json = json.load(f)

    key_map = {}
    for case in key_json:
        case_id = case["case_id"]
        gold = {}
        for ans in case.get("answers", []):
            gold[ans["sentence_id"]] = ans["relevance"]
        key_map[case_id] = gold

    # -----------------------
    # Load XML
    # -----------------------

    tree = ET.parse(xml_path)
    root = tree.getroot()

    cases = []

    for case_elem in root.findall("case"):
        case_id = case_elem.attrib["id"]

        clinician_question_elem = case_elem.find("clinician_question")
        clinician_question = clinician_question_elem.text.strip()

        sentences = {}
        sent_block = case_elem.find("note_excerpt_sentences")
        for sent in sent_block.findall("sentence"):
            sent_id = sent.attrib["id"]
            sent_text = "".join(sent.itertext()).strip()
            sentences[sent_id] = sent_text

        cases.append({
            "case_id": case_id,
            "clinician_question": clinician_question,
            "sentences": sentences,
            "gold_labels": key_map.get(case_id, {})
        })

    return cases
