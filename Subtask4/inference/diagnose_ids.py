"""
Run from your subtask4 working directory to diagnose the ID mismatch.

Usage:
    python diagnose_ids.py
"""

import json
import xml.etree.ElementTree as ET


def main():
    xml_file = "../../data/dev/archehr-qa.xml"
    json_file = "../../data/dev/archehr-qa_key.json"

    # --- XML side ---
    tree = ET.parse(xml_file)
    root = tree.getroot()

    xml_cases = {}
    for case in root.findall("case"):
        case_id = case.get("id")
        sentences_node = case.find("note_excerpt_sentences")
        if sentences_node is None:
            continue
        sents = []
        for sent in sentences_node.findall("sentence"):
            sid = sent.get("id")
            text = (sent.text or "").strip()
            sents.append({"id": sid, "text": text[:80]})
        xml_cases[case_id] = sents

    # --- JSON side ---
    with open(json_file, "r") as f:
        json_data = json.load(f)

    json_cases = {}
    for item in json_data:
        case_id = item.get("case_id")
        # Print ALL top-level keys so we know what fields exist
        if case_id == "1":
            print("=" * 60)
            print(f"JSON top-level keys for case 1: {list(item.keys())}")
            print("=" * 60)

        # Try "answers" field (used by scoring script for valid_evidence_ids)
        answers = item.get("answers", [])
        json_cases[case_id] = {
            "answers": [
                {"sentence_id": a.get("sentence_id"), "text": a.get("text", "")[:80]}
                for a in answers
            ],
            "citations": [],
        }

        # Also extract gold citations from clinician_answer_sentences
        for ans_sent in item.get("clinician_answer_sentences", []):
            json_cases[case_id]["citations"].append({
                "answer_id": ans_sent.get("id"),
                "citation_ids": ans_sent.get("citations", []),
            })

    # --- Compare for first 3 cases ---
    for case_id in ["1", "2", "3"]:
        print(f"\n{'=' * 60}")
        print(f"CASE {case_id}")
        print(f"{'=' * 60}")

        xml_sents = xml_cases.get(case_id, [])
        json_data = json_cases.get(case_id, {})
        json_answers = json_data.get("answers", [])
        citations = json_data.get("citations", [])

        print(f"\nXML note sentences (first 5):")
        for s in xml_sents[:5]:
            print(f"  id={s['id']:>3s}  | {s['text']}")

        print(f"\nJSON 'answers' field (first 5):")
        if json_answers:
            for s in json_answers[:5]:
                print(f"  sentence_id={s['sentence_id']:>3s}  | {s['text'][:80]}")
        else:
            print("  (EMPTY - 'answers' field not found or empty)")

        print(f"\nGold citations (from clinician_answer_sentences):")
        for c in citations:
            print(f"  answer {c['answer_id']} -> note sentences {c['citation_ids']}")

        # Check offset
        if xml_sents and json_answers:
            xml_first = xml_sents[0]
            json_first = json_answers[0]
            if xml_first["text"][:40] == json_first["text"][:40]:
                offset = int(xml_first["id"]) - int(json_first["sentence_id"])
                print(f"\n  >>> SAME first sentence text. ID offset: XML {xml_first['id']} vs JSON {json_first['sentence_id']} = {offset:+d}")
            else:
                print(f"\n  >>> DIFFERENT first sentence text!")
                print(f"      XML[0]: {xml_first['text'][:60]}")
                print(f"      JSON[0]: {json_first['text'][:60]}")


if __name__ == "__main__":
    main()