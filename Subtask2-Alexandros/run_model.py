import json
import sys
from pathlib import Path
import re

CURRENT_DIR = Path(__file__).resolve().parent
COMMON_PATH = CURRENT_DIR.parent / "ArchEHR-QA-2026" / "common"
sys.path.append(str(COMMON_PATH))

from common.providers import CloudProvider
from load_archehr import load_all_cases


MODEL_NAME = sys.argv[1]

XML_PATH = "/ceph/home/student.aau.dk/lj02sb/ArchEHR/atificial_data/v1.4/dev/archehr-qa.xml"
KEY_PATH = "/ceph/home/student.aau.dk/lj02sb/ArchEHR/atificial_data/v1.4/dev/archehr-qa_key.json"

OUT_PATH = f"outputs/{MODEL_NAME.replace('/', '_')}.json"


def extract_json(text):
    """
    Extract first JSON object from model output safely.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def build_messages(case_id, question, sentences_block):
    return [
        {
            "role": "system",
            "content": (
                "You are a clinical evidence identification expert.\n"
                "Select ALL sentence IDs that help answer the clinician's question.\n"
                "Prefer recall over precision.\n"
                "Return ONLY a JSON object:\n"
                "{\n"
                f'  "case_id": "{case_id}",\n'
                '  "prediction": ["1", "2"]\n'
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Clinician question:\n{question}\n\n"
                f"Sentences:\n{sentences_block}"
            )
        }
    ]


def main():
    provider = CloudProvider(model_name=MODEL_NAME)
    cases = load_all_cases(XML_PATH, KEY_PATH)

    submission = []

    for case in cases:
        case_id = case["case_id"]
        question = case["clinician_question"]
        sentences = case["sentences"]

        valid_ids = set(sentences.keys())

        numbered_sentences = "\n".join(
            [f"{sid}. {text}" for sid, text in sentences.items()]
        )

        messages = build_messages(case_id, question, numbered_sentences)

        try:
            result = provider.generate(messages)
            content = result["content"]

            json_text = extract_json(content)

            if json_text:
                parsed = json.loads(json_text)
                prediction = parsed.get("prediction", [])
            else:
                prediction = []

        except Exception as e:
            print("Error:", e)
            prediction = []

        # Ensure string IDs
        prediction = [str(p) for p in prediction]

        # Remove invalid IDs
        prediction = [p for p in prediction if p in valid_ids]

        # Sort numerically
        prediction = sorted(prediction, key=int)

        submission.append({
            "case_id": case_id,
            "prediction": prediction
        })

        print(f"{MODEL_NAME} → finished case {case_id}")

    Path("outputs").mkdir(exist_ok=True)

    with open(OUT_PATH, "w") as f:
        json.dump(submission, f, indent=2)

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
