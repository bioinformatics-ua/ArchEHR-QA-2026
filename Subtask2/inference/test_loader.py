from pathlib import Path
from dataloader import ArchEHRSubtask2DataLoader

loader = ArchEHRSubtask2DataLoader(
    Path("../../data/dev/archehr-qa.xml")
)

data = loader.load()

print(f"Loaded {len(data)} cases")
print(f"First case keys: {data[0].keys()}")
print(f"Sentences in first case: {len(data[0]['sentences'])}")
