from pathlib import Path
from dataloader import ArchEHRSubtask4DataLoader


if __name__ == "__main__":
    xml_path = Path("../data/dev/archehr-qa.xml")
    key_path = Path("../data/dev/archehr-qa_key.json")

    loader = ArchEHRSubtask4DataLoader(
        xml_path=xml_path,
        key_json_path=key_path,
    )

    cases = loader.load()

    print(f"Loaded {len(cases)} cases\n")

    first_case = cases[0]

    print("Case ID:", first_case["case_id"])
    print("Answer sentences:", len(first_case["answer_sentences"]))
    print("Note sentences:", len(first_case["note_sentences"]))
    print("Gold alignments:", first_case["gold_alignments"])