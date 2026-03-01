import json
from collections import Counter, defaultdict
from pathlib import Path

FILES = [
    "../outputs/dev/run_t00.json",
    "../outputs/dev/run_t02.json",
    "../outputs/dev/run_t04.json",
]

OUTPUT = "../outputs/dev/aggregated_majority.json"

def load(path):
    with open(path) as f:
        return json.load(f)

runs = [load(p) for p in FILES]

final = []

for case_idx in range(len(runs[0])):
    case_id = runs[0][case_idx]["case_id"]
    answers = runs[0][case_idx]["prediction"]

    aggregated_predictions = []

    for ans_idx in range(len(answers)):
        answer_id = answers[ans_idx]["answer_id"]

        all_evidence = []

        for run in runs:
            ev = run[case_idx]["prediction"][ans_idx]["evidence_id"]
            all_evidence.extend(ev)

        counter = Counter(all_evidence)

        # Majority rule (appears in at least 2 runs)
        majority = [e for e, count in counter.items() if count >= 2]

        # Fallback: if no majority, use union (safe recall boost)
        if not majority:
            majority = list(set(all_evidence))

        aggregated_predictions.append({
            "answer_id": answer_id,
            "evidence_id": sorted(majority, key=lambda x: int(x))
        })

    final.append({
        "case_id": case_id,
        "prediction": aggregated_predictions
    })

with open(OUTPUT, "w") as f:
    json.dump(final, f, indent=4)

print(f"Saved aggregated file to {OUTPUT}")