import json
import subprocess
import itertools
from pathlib import Path

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
OUTPUT_DIR = Path("../outputs/dev")
RESULTS_DIR = Path("../results/dev")
SEARCH_OUTPUT_DIR = Path("../outputs/dev/search_combos")
SEARCH_RESULTS_DIR = Path("../results/dev/search_combos")
KEY_PATH = Path("../../data-subtask2&3/dev/archehr-qa_key.json")
SCORING_SCRIPT = Path("../evaluation/scoring_subtask_4.py")

CANDIDATE_FILES = [
    "google-gemini-2-5-flash_prompt_5.json",
    "anthropic-claude-opus-4-5_prompt_5.json",
    "anthropic-claude-opus-4-6_prompt_5.json",
    "x-ai-grok-4-fast_prompt_5.json",
    "anthropic-claude-sonnet-4-5_prompt_5.json",
    "google-gemini-3-flash-preview_prompt_5.json",
    "anthropic-claude-sonnet-4_prompt_5.json",
    "x-ai-grok-4-1-fast_prompt_5.json",
    "nvidia-nemotron-3-nano-30b-a3b:free_prompt_5.json",
    "deepseek-deepseek-v3-2_prompt_5.json",
    "openai-gpt-5_prompt_5.json",
    "google-gemini-2-0-flash-001_prompt_5.json",
    "openai-gpt-4-1_prompt_5.json",
    "qwen-qwen3-max-thinking_prompt_5.json",
    "qwen-qwen3-5-plus-02-15_prompt_5.json",
]

ENSEMBLE_STRATEGY = "majority"
MAJORITY_THRESHOLD = 2
MIN_COMBO_SIZE = 3   # minimum number of models in a combination
MAX_COMBO_SIZE = 7   # maximum number of models in a combination (keep manageable)

# ----------------------------------------
# HELPERS
# ----------------------------------------

def run_ensemble(combo, output_file):
    input_paths = [str(OUTPUT_DIR / f) for f in combo]
    cmd = [
        "uv", "run", "python", "ensemble.py",
        "--inputs", *input_paths,
        "--strategy", ENSEMBLE_STRATEGY,
        "--majority-threshold", str(MAJORITY_THRESHOLD),
        "--output", str(SEARCH_OUTPUT_DIR / output_file),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def run_scoring(output_file):
    result_file = SEARCH_RESULTS_DIR / output_file
    cmd = [
        "uv", "run", "python", str(SCORING_SCRIPT),
        "--submission_path", str(SEARCH_OUTPUT_DIR / output_file),
        "--key_path", str(KEY_PATH),
        "--out_file_path", str(result_file),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    with open(result_file) as f:
        result = json.load(f)

    return result.get("overall_score")


# ----------------------------------------
# MAIN SEARCH
# ----------------------------------------
if __name__ == "__main__":
    # Create search combo directories if they don't exist
    SEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to only existing files
    available = [f for f in CANDIDATE_FILES if (OUTPUT_DIR / f).exists()]
    print(f"Found {len(available)} available output files")

    results = []
    total = sum(len(list(itertools.combinations(available, r))) for r in range(MIN_COMBO_SIZE, MAX_COMBO_SIZE + 1))
    print(f"Testing {total} combinations (size {MIN_COMBO_SIZE}-{MAX_COMBO_SIZE})...\n")

    count = 0
    for size in range(MIN_COMBO_SIZE, MAX_COMBO_SIZE + 1):
        for combo in itertools.combinations(available, size):
            count += 1
            combo_name = f"search_combo_{count}.json"
            short_names = [f.replace("_prompt_5.json", "") for f in combo]

            try:
                run_ensemble(combo, combo_name)
                score = run_scoring(combo_name)
                results.append((score, list(combo)))
                print(f"[{count}/{total}] Score: {score:.4f} | {short_names}")
            except Exception as e:
                print(f"[{count}/{total}] FAILED: {short_names} -> {e}")

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n========================================")
    print("TOP 10 BEST COMBINATIONS:")
    print("========================================")
    for i, (score, combo) in enumerate(results[:10], 1):
        short = [f.replace("_prompt_5.json", "") for f in combo]
        print(f"\n#{i} Score: {score:.4f}")
        for m in short:
            print(f"   - {m}")

    # Save full results
    with open("search_results.json", "w") as f:
        json.dump([{"score": s, "combo": c} for s, c in results], f, indent=2)
    print("\nFull results saved to search_results.json")