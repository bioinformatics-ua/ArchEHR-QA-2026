from pathlib import Path
from dataloader import ArchEHRSubtask4DataLoader
from retriever import SBertRetriever
from sentence_transformers import SentenceTransformer
from cross_encoder_model import CrossEncoderClassifier


if __name__ == "__main__":

    xml_path = Path("../data/dev/archehr-qa.xml")
    key_path = Path("../data/dev/archehr-qa_key.json")

    loader = ArchEHRSubtask4DataLoader(xml_path, key_path)
    cases = loader.load()

    case = cases[0]
    answer = case["answer_sentences"][0]

    retriever = SBertRetriever()
    cross_encoder = CrossEncoderClassifier(threshold=0.5)

    ranked = retriever.rank(
        answer_text=answer["text"],
        note_sentences=case["note_sentences"],
        top_k=5,
    )

    # Dynamic filtering
    top_score = ranked[0]["score"]
    margin = 0.10

    filtered = [
        r for r in ranked
        if r["score"] >= top_score - margin
    ]

    results = cross_encoder.classify(
        answer_text=answer["text"],
        candidate_notes=filtered,
    )

    print("\nFinal decisions:\n")

    for r in results:
        print(
            f"Note ID: {r['note_id']} | "
            f"Cross score: {r['score']:.4f} | "
            f"Support: {r['support']}"
        )