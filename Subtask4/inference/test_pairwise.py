from pathlib import Path
from dataloader import ArchEHRSubtask4DataLoader
from pairwise import build_pairwise_matrix


if __name__ == "__main__":
    xml_path = Path("../data/dev/archehr-qa.xml")
    key_path = Path("../data/dev/archehr-qa_key.json")

    loader = ArchEHRSubtask4DataLoader(xml_path, key_path)
    cases = loader.load()

    case = cases[0]

    pairs = build_pairwise_matrix(case)

    print("Total pairs:", len(pairs))

    positives = [p for p in pairs if p["label"] == 1]
    print("Positive pairs:", len(positives))

    print("\nExample positive:")
    print(positives[0])