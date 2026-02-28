from pathlib import Path
from dataloader import ArchEHRSubtask4DataLoader
from retriever import SBertRetriever


if __name__ == "__main__":
    xml_path = Path("../data/dev/archehr-qa.xml")
    key_path = Path("../data/dev/archehr-qa_key.json")

    loader = ArchEHRSubtask4DataLoader(xml_path, key_path)
    cases = loader.load()

    case = cases[0]
    answer = case["answer_sentences"][0]

    retriever = SBertRetriever()

    ranked = retriever.rank(
        answer_text=answer["text"],
        note_sentences=case["note_sentences"],
        top_k=5,
    )

    print("Answer:", answer["text"])
    print("\nTop-5 retrieved notes:\n")

    for r in ranked:
        print(f"Score: {r['score']:.4f} | ID: {r['note_id']}")
        print(r["note_text"])
        print("-" * 50)