import json
from collections import defaultdict

GOLD_PATH = "../data/dev/archehr-qa_key.json"
PRED_PATH = "../outputs/dev/google-gemini-2-5-flash_prompt_5.json"

with open(GOLD_PATH, "r") as f:
    gold_list = json.load(f)

with open(PRED_PATH, "r") as f:
    pred_list = json.load(f)

gold_data = {case["case_id"]: case for case in gold_list}
pred_data = {case["case_id"]: case for case in pred_list}

error_summary = defaultdict(int)

for case_id in gold_data:
    gold_case = gold_data[case_id]
    pred_case = pred_data.get(case_id)

    if not pred_case:
        print(f"Missing prediction for case {case_id}")
        continue

    # GOLD MAP
    gold_map = {}
    for sent in gold_case.get("clinician_answer_sentences", []):
        ans_id = sent["id"]
        citations = sent.get("citations", [])
        gold_map[ans_id] = set(citations)

    # PRED MAP
    pred_map = {}
    for item in pred_case.get("prediction", []):
        ans_id = item["answer_id"]
        pred_map[ans_id] = set(item.get("evidence_id", []))

    for ans_id in gold_map:
        gold_set = gold_map[ans_id]
        pred_set = pred_map.get(ans_id, set())

        if gold_set == pred_set:
            continue

        if gold_set == set() and pred_set != set():
            error_summary["False Positive (should be empty)"] += 1

        elif gold_set != set() and pred_set == set():
            error_summary["False Empty"] += 1

        elif len(pred_set) > len(gold_set):
            error_summary["Over-citation"] += 1

        elif len(pred_set) < len(gold_set):
            error_summary["Under-citation"] += 1

        else:
            error_summary["Mismatch (different sentences)"] += 1

print("\nERROR SUMMARY")
for k, v in error_summary.items():
    print(f"{k}: {v}")