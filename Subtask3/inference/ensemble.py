"""
Subtask 3 — LLM-as-Judge Ensemble
==================================
Given N candidate prediction files (from different models / prompts),
use a strong judge model to select or synthesise the best answer per case.

Usage:
    python ensemble.py \
        --xml-file ../../data/test/archehr-qa.xml \
        --candidate-files ../outputs/test/anthropic-claude-sonnet-4-5_prompt_3.json \
                          ../outputs/test/google-gemini-2-5-flash_prompt_9.json \
                          ... \
        --judge-model anthropic/claude-sonnet-4.5 \
        --output-file ../outputs/test/ensemble_judge.json \
        --mode select          # select | merge
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SELECT_SYSTEM = (
    "You are an expert clinical evaluator. Your job is to choose the single "
    "best answer from a numbered list of candidate answers to a patient's "
    "medical question, based on a clinical note excerpt."
)

SELECT_USER = """\
### Patient Question
{patient_question}

### Clinician-Interpreted Question
{clinician_question}

### Clinical Note Excerpt (numbered sentences)
{sentences}

### Candidate Answers
{candidates}

### Selection Criteria (in priority order)
1. **Faithfulness** — every claim must be supported by the numbered sentences.
2. **Medical completeness** — covers the key clinical information that answers the question.
3. **Terminology** — retains exact medical terms, procedure names, and diagnoses from the note.
4. **Conciseness** — no filler, hedging, or generic statements; maximum 75 words.

Reply with ONLY the number of the best candidate (e.g. "3"). Nothing else."""

MERGE_SYSTEM = (
    "You are an expert clinical writer. Your job is to synthesise the best "
    "possible answer to a patient's medical question by combining the "
    "strongest elements from multiple candidate answers, grounded in the "
    "clinical note excerpt."
)

MERGE_USER = """\
### Patient Question
{patient_question}

### Clinician-Interpreted Question
{clinician_question}

### Clinical Note Excerpt (numbered sentences)
{sentences}

### Candidate Answers
{candidates}

### Instructions
- Produce ONE answer of 3-5 sentences, maximum 75 words.
- Ground every claim in the numbered sentences above.
- Retain exact medical terminology from the note.
- Do NOT add external knowledge or speculation.
- Prioritise the most clinically relevant information across all candidates.
- Output ONLY the final answer paragraph — no preamble, no reasoning.

Answer:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_candidates(
    candidate_files: list[Path],
) -> dict[str, list[tuple[str, str]]]:
    """Return {case_id: [(label, prediction), ...]} for all candidate files."""
    candidates: dict[str, list[tuple[str, str]]] = {}
    for fpath in candidate_files:
        label = fpath.stem  # e.g. "anthropic-claude-sonnet-4-5_prompt_3"
        data = json.loads(fpath.read_text())
        for entry in data:
            cid = str(entry["case_id"])
            pred = entry.get("prediction", "").strip()
            candidates.setdefault(cid, []).append((label, pred))
    return candidates


def load_cases(xml_path: Path) -> dict[str, dict]:
    """Load XML cases into {case_id: {...}} via the existing dataloader."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataloader import ArchEHRSubtask3DataLoader

    cases_list = ArchEHRSubtask3DataLoader(xml_path).load()
    return {str(c["case_id"]): c for c in cases_list}


def format_sentences(case: dict) -> str:
    return "\n".join(
        f"{s['sentence_id']}: {s['text']}" for s in case["sentences"]
    )


def format_candidates(pairs: list[tuple[str, str]]) -> str:
    """Number candidates 1..N, skipping empty predictions."""
    lines = []
    for i, (_label, pred) in enumerate(pairs, 1):
        if pred:
            lines.append(f"[{i}] {pred}")
    return "\n\n".join(lines)


def build_judge_prompt(
    case: dict,
    pairs: list[tuple[str, str]],
    mode: str,
) -> list[dict]:
    sentences_text = format_sentences(case)
    candidates_text = format_candidates(pairs)

    if mode == "select":
        system_msg = SELECT_SYSTEM
        user_msg = SELECT_USER
    else:
        system_msg = MERGE_SYSTEM
        user_msg = MERGE_USER

    user_content = user_msg.format(
        patient_question=case["patient_question"],
        clinician_question=case["clinician_question"],
        sentences=sentences_text,
        candidates=candidates_text,
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def parse_selection(response: str, n_candidates: int) -> int | None:
    """Extract selected candidate index (0-based) from judge response."""
    import re

    # Look for a standalone number
    match = re.search(r"\b(\d+)\b", response.strip())
    if match:
        idx = int(match.group(1)) - 1  # convert to 0-based
        if 0 <= idx < n_candidates:
            return idx
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Subtask 3: LLM-as-Judge Ensemble"
    )
    parser.add_argument(
        "--xml-file", type=Path, required=True, help="Path to archehr-qa.xml"
    )
    parser.add_argument(
        "--candidate-files",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction JSON files to ensemble",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="OpenRouter model to use as judge",
    )
    parser.add_argument(
        "--mode",
        choices=["select", "merge"],
        default="select",
        help="select: pick best candidate; merge: synthesise new answer",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output submission JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge model temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for judge response",
    )
    parser.add_argument(
        "--fallback",
        type=int,
        default=0,
        help="0-based index of fallback candidate file if judge fails (default: first file)",
    )
    args = parser.parse_args()

    # --- Setup OpenRouter client ---
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY or OPENAI_API_KEY env variable is required"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://archehr-qa",
            "X-Title": "ArchEHR-QA-Subtask3-Ensemble",
            "Accept": "application/json",
        },
    )

    # --- Load data ---
    print(f"Loading XML cases from {args.xml_file}...")
    cases = load_cases(args.xml_file)
    print(f"Loaded {len(cases)} cases.")

    print(f"Loading {len(args.candidate_files)} candidate files...")
    for f in args.candidate_files:
        print(f"  - {f.name}")
    candidates = load_candidates(args.candidate_files)
    print(f"Found candidates for {len(candidates)} case IDs.")

    # --- Run judge ---
    MAX_WORDS = 75
    results = []
    analysis = []
    total_cost = 0.0

    sorted_case_ids = sorted(candidates.keys(), key=lambda x: int(x))

    for i, case_id in enumerate(sorted_case_ids):
        case = cases.get(case_id)
        if case is None:
            print(f"  [WARN] case {case_id} not in XML, skipping.")
            continue

        pairs = candidates[case_id]

        # Skip cases where all candidates are empty
        non_empty = [(label, pred) for label, pred in pairs if pred.strip()]
        if not non_empty:
            print(f"  [case {case_id}] all candidates empty, using empty.")
            results.append({"case_id": case_id, "prediction": ""})
            analysis.append(
                {
                    "case_id": case_id,
                    "selected": None,
                    "judge_raw": "",
                    "final_answer": "",
                }
            )
            continue

        # If only one non-empty candidate, just use it
        if len(non_empty) == 1:
            answer = non_empty[0][1]
            words = answer.split()
            if len(words) > MAX_WORDS:
                answer = " ".join(words[:MAX_WORDS])
            results.append({"case_id": case_id, "prediction": answer})
            analysis.append(
                {
                    "case_id": case_id,
                    "selected": non_empty[0][0],
                    "judge_raw": "only_one_candidate",
                    "final_answer": answer,
                }
            )
            continue

        # Build prompt and call judge
        prompt = build_judge_prompt(case, pairs, args.mode)

        try:
            response = client.chat.completions.create(
                model=args.judge_model,
                messages=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            judge_text = (response.choices[0].message.content or "").strip()

            # Track cost
            if response.usage and hasattr(response.usage, "cost"):
                cost = getattr(response.usage, "cost", 0) or 0
                total_cost += cost

            if args.mode == "select":
                idx = parse_selection(judge_text, len(pairs))
                if idx is not None:
                    answer = pairs[idx][1]
                    selected_label = pairs[idx][0]
                else:
                    # Fallback: use the configured fallback candidate
                    fallback_idx = min(args.fallback, len(pairs) - 1)
                    answer = pairs[fallback_idx][1]
                    selected_label = f"FALLBACK:{pairs[fallback_idx][0]}"
                    print(
                        f"  [case {case_id}] judge returned unparseable "
                        f"'{judge_text[:60]}', using fallback."
                    )
            else:
                # merge mode — judge wrote the answer directly
                answer = judge_text
                selected_label = "MERGED"

        except Exception as e:
            print(f"  [case {case_id}] judge error: {e}, using fallback.")
            fallback_idx = min(args.fallback, len(pairs) - 1)
            answer = pairs[fallback_idx][1]
            selected_label = f"ERROR:{pairs[fallback_idx][0]}"
            judge_text = str(e)

        # Enforce 75-word limit
        words = answer.split()
        if len(words) > MAX_WORDS:
            answer = " ".join(words[:MAX_WORDS])

        results.append({"case_id": case_id, "prediction": answer})
        analysis.append(
            {
                "case_id": case_id,
                "selected": selected_label,
                "judge_raw": judge_text,
                "final_answer": answer,
                "word_count": len(answer.split()),
            }
        )

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(sorted_case_ids):
            print(
                f"  [{i + 1}/{len(sorted_case_ids)}] "
                f"case {case_id} -> {selected_label}"
            )

        # Tiny delay to avoid rate limits
        time.sleep(0.1)

    # --- Write submission JSON ---
    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Write analysis JSON ---
    base_dir = Path(__file__).resolve().parent.parent
    split = output_path.parent.name
    fname = output_path.name
    analysis_file = base_dir / "analysis" / split / fname
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    print(
        f"\n[DONE] Ensemble complete.\n"
        f"  Mode:       {args.mode}\n"
        f"  Judge:      {args.judge_model}\n"
        f"  Candidates: {len(args.candidate_files)}\n"
        f"  Cases:      {len(results)}\n"
        f"  Submission: {output_path}\n"
        f"  Analysis:   {analysis_file}\n"
        f"  Est. cost:  ${total_cost:.4f}"
    )


if __name__ == "__main__":
    main()
