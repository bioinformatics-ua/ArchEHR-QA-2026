import json
import subprocess
import itertools
import sys
from pathlib import Path

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "default"

OUTPUT_DIR = Path("../outputs/dev")
RESULTS_DIR = Path("../results/dev")
SEARCH_OUTPUT_DIR = Path(f"../outputs/dev/search_combos/{RUN_ID}")
SEARCH_RESULTS_DIR = Path(f"../results/dev/search_combos/{RUN_ID}")
KEY_PATH = Path("../../data-subtask2&3/dev/archehr-qa_key.json")
SCORING_SCRIPT = Path("../evaluation/scoring_subtask_4.py")

CANDIDATE_FILES = [

    "google-gemini-2-5-flash_prompt_5.json",        # 89.78
    "google-gemini-2-5-flash_prompt_6.json",        # 89.55
    "anthropic-claude-opus-4-5_prompt_5.json",      # 89.37
    "google-gemini-2-5-flash_prompt_7.json",        # 89.12
    "google-gemini-2-0-flash-001_prompt_5.json",    # 89.05
    "openai-gpt-4-1_prompt_6.json",                 # 88.97
    "x-ai-grok-4-1-fast_prompt_7.json",             # 88.97
    "anthropic-claude-opus-4-6_prompt_5.json",      # 88.72
    "openai-gpt-4-1_prompt_5.json",                 # 88.14
    "openai-gpt-5_prompt_7.json",                   # 87.91
    "x-ai-grok-4-fast_prompt_5.json",               # 87.40
    "google-gemini-3-flash-preview_prompt_6.json",  # 87.28
    "anthropic-claude-opus-4-5_prompt_7.json",      # 87.23
    "openai-gpt-4-1_prompt_7.json",                 # 87.10
    "anthropic-claude-sonnet-4-5_prompt_5.json",    # 86.99

    "anthropic-claude-opus-4-6_prompt_7.json",      # High recall
    "anthropic-claude-sonnet-4_prompt_6.json",      # High precision
    "anthropic-claude-sonnet-4-6_prompt_7.json",    # High recall
    "anthropic-claude-sonnet-4-5_prompt_6.json",    # High precision
    "qwen-qwen3-max-thinking_prompt_6.json",        # High precision
    "x-ai-grok-4-fast_prompt_6.json",               # High precision
    "x-ai-grok-4-1-fast_prompt_6.json",             # High precision
    "google-gemini-3-flash-preview_prompt_7.json",  # High recall
    "google-gemini-2-0-flash-001_prompt_6.json",    # High precision

    # "google-gemini-2-5-flash_prompt_5.json",
    # "google-gemini-2-5-flash_prompt_6.json",
    # "google-gemini-2-5-flash_prompt_7.json",
    # "anthropic-claude-opus-4-5_prompt_5.json",
    # "anthropic-claude-opus-4-5_prompt_6.json",
    # "anthropic-claude-opus-4-5_prompt_7.json",
    # "anthropic-claude-opus-4-6_prompt_5.json",
    # "anthropic-claude-opus-4-6_prompt_6.json",
    # "anthropic-claude-opus-4-6_prompt_7.json",
    # "x-ai-grok-4-fast_prompt_5.json",
    # "x-ai-grok-4-fast_prompt_6.json",
    # "anthropic-claude-sonnet-4-5_prompt_5.json",
    # "anthropic-claude-sonnet-4-5_prompt_6.json",
    # "anthropic-claude-sonnet-4-5_prompt_7.json",
    # "anthropic-claude-sonnet-4-6_prompt_7.json",
    # "google-gemini-3-flash-preview_prompt_5.json",
    # "google-gemini-3-flash-preview_prompt_6.json",
    # "google-gemini-3-flash-preview_prompt_7.json",
    # "anthropic-claude-sonnet-4_prompt_5.json",
    # "anthropic-claude-sonnet-4_prompt_6.json",
    # "anthropic-claude-sonnet-4_prompt_7.json",
    # "x-ai-grok-4-1-fast_prompt_5.json",
    # "x-ai-grok-4-1-fast_prompt_6.json",
    # "x-ai-grok-4-1-fast_prompt_7.json",
    # "nvidia-nemotron-3-nano-30b-a3b:free_prompt_5.json",
    # "deepseek-deepseek-v3-2_prompt_5.json",
    # "openai-gpt-5_prompt_5.json",
    # "openai-gpt-5_prompt_6.json",
    # "openai-gpt-5_prompt_7.json",
    # "google-gemini-2-0-flash-001_prompt_5.json",
    # "google-gemini-2-0-flash-001_prompt_6.json",
    # "google-gemini-2-0-flash-001_prompt_7.json",
    # "openai-gpt-4-1_prompt_5.json",
    # "openai-gpt-4-1_prompt_6.json",
    # "openai-gpt-4-1_prompt_7.json",
    # "qwen-qwen3-max-thinking_prompt_5.json",
    # "qwen-qwen3-max-thinking_prompt_6.json",
    # "qwen-qwen3-5-plus-02-15_prompt_5.json",
]

ENSEMBLE_STRATEGY = "majority"
MAJORITY_THRESHOLD = 2
MIN_COMBO_SIZE = 3
MAX_COMBO_SIZE = 4

# Phase 2: how many top combos to test fallback models on
TOP_N_FOR_FALLBACK = 50

# ----------------------------------------
# HELPERS
# ----------------------------------------

def run_ensemble(combo, output_file, fallback=None):
    input_paths = [str(OUTPUT_DIR / f) for f in combo]
    cmd = [
        "uv", "run", "python", "ensemble.py",
        "--inputs", *input_paths,
        "--strategy", ENSEMBLE_STRATEGY,
        "--majority-threshold", str(MAJORITY_THRESHOLD),
        "--output", str(SEARCH_OUTPUT_DIR / output_file),
    ]
    if fallback:
        cmd += ["--fallback", str(OUTPUT_DIR / fallback)]
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
    print(f"Run ID: {RUN_ID}")

    SEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    available = [f for f in CANDIDATE_FILES if (OUTPUT_DIR / f).exists()]
    print(f"Found {len(available)} available output files")

    # ----------------------------------------
    # PHASE 1: Find best combinations (no fallback)
    # ----------------------------------------
    total = sum(len(list(itertools.combinations(available, r))) for r in range(MIN_COMBO_SIZE, MAX_COMBO_SIZE + 1))
    print(f"\n{'='*40}")
    print(f"PHASE 1: Testing {total} combinations (no fallback, size {MIN_COMBO_SIZE}-{MAX_COMBO_SIZE})...")
    print(f"{'='*40}\n")

    phase1_results = []
    count = 0
    for size in range(MIN_COMBO_SIZE, MAX_COMBO_SIZE + 1):
        for combo in itertools.combinations(available, size):
            count += 1
            combo_name = f"search_combo_{count}.json"
            short_names = [f.replace("_prompt_5.json", "") for f in combo]

            try:
                run_ensemble(combo, combo_name)
                score = run_scoring(combo_name)
                phase1_results.append((score, list(combo)))
                print(f"[{count}/{total}] Score: {score:.4f} | {short_names}")
            except Exception as e:
                print(f"[{count}/{total}] FAILED: {short_names} -> {e}")

    # Sort phase 1 results
    phase1_results.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTOP {TOP_N_FOR_FALLBACK} COMBOS FROM PHASE 1:")
    for i, (score, combo) in enumerate(phase1_results[:TOP_N_FOR_FALLBACK], 1):
        short = [f.replace("_prompt_5.json", "") for f in combo]
        print(f"  #{i} Score: {score:.4f} | {short}")

    # Save phase 1 results
    with open(RESULTS_DIR / f"search_results_phase1_{RUN_ID}.json", "w") as f:
        json.dump(
            [{"score": s, "combo": c} for s, c in phase1_results],
            f, indent=2
        )
    print(f"\nPhase 1 results saved to search_results_phase1_{RUN_ID}.json")

    # ----------------------------------------
    # PHASE 2: Find best fallback for top N combos
    # ----------------------------------------
    top_combos = [combo for _, combo in phase1_results[:TOP_N_FOR_FALLBACK]]
    fallback_candidates = available + [None]  # None = no fallback
    phase2_total = len(top_combos) * len(fallback_candidates)

    print(f"\n{'='*40}")
    print(f"PHASE 2: Testing {phase2_total} fallback combinations on top {TOP_N_FOR_FALLBACK} combos...")
    print(f"{'='*40}\n")

    phase2_results = []
    count = 0
    for combo in top_combos:
        for fallback in fallback_candidates:
            count += 1
            fallback_tag = fallback.replace("_prompt_5.json", "") if fallback else "none"
            combo_name = f"search_combo_p2_{count}.json"
            short_names = [f.replace("_prompt_5.json", "") for f in combo]

            try:
                run_ensemble(combo, combo_name, fallback=fallback)
                score = run_scoring(combo_name)
                phase2_results.append((score, list(combo), fallback))
                print(f"[{count}/{phase2_total}] Score: {score:.4f} | fallback={fallback_tag} | {short_names}")
            except Exception as e:
                print(f"[{count}/{phase2_total}] FAILED: fallback={fallback_tag} | {short_names} -> {e}")

    # Sort phase 2 results
    phase2_results.sort(key=lambda x: x[0], reverse=True)

    # ----------------------------------------
    # FINAL RESULTS
    # ----------------------------------------
    print("\n========================================")
    print("TOP 10 BEST COMBINATIONS (with fallback):")
    print("========================================")
    for i, (score, combo, fallback) in enumerate(phase2_results[:10], 1):
        short = [f.replace("_prompt_5.json", "") for f in combo]
        fallback_tag = fallback.replace("_prompt_5.json", "") if fallback else "none"
        print(f"\n#{i} Score: {score:.4f} | fallback={fallback_tag}")
        for m in short:
            print(f"   - {m}")

    # Save phase 2 results
    with open(RESULTS_DIR / f"search_results_phase2_{RUN_ID}.json", "w") as f:
        json.dump(
            [{"score": s, "combo": c, "fallback": fb} for s, c, fb in phase2_results],
            f, indent=2
        )
    print(f"Phase 2 results saved to search_results_phase2_{RUN_ID}.json")