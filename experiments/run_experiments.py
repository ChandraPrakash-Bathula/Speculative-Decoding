"""
Headless experiment harness for the speculative decoding laboratory.

Runs a matrix of (prompt x speculation length x repeat), records every raw
measurement to JSON, and never aggregates anything here: aggregation lives in
analyze.py so that the raw records stay auditable and any published number can
be traced back to the run that produced it.

Usage:
    python experiments/run_experiments.py                  # full matrix
    python experiments/run_experiments.py --quick          # small smoke matrix
    python experiments/run_experiments.py --max-new-tokens 96 --repeats 5

Run this on the machine whose numbers you intend to publish. The draft/target
cost ratio c is a property of the hardware, not of the model pair, so results
from one machine do not transfer to another.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from backend.main import (  # noqa: E402
    DEVICE,
    DTYPE,
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    load_models,
    run_baseline,
    run_speculative,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Prompts deliberately span genres, because acceptance rate tracks how
# predictable the continuation is: boilerplate definitions are easy for a small
# draft model, open-ended reasoning is not.
PROMPTS = {
    "definition": "Explain what a hash map is and give a short example.",
    "code": "Write a Python function that reverses a linked list.",
    "explanation": "Summarise why transformers replaced RNNs for sequence modelling.",
    "list": "List five practical uses of binary search trees.",
    "reasoning": "A train leaves at 3pm going 60mph and another at 4pm going 80mph. When does the second catch the first? Reason step by step.",
    "creative": "Write an opening paragraph for a story about a lighthouse keeper.",
}

K_VALUES = [1, 2, 3, 4, 6, 8]


def build_record(kind, prompt_id, prompt, k, repeat, payload):
    """Strip the generated text down to a hash plus a preview.

    Full text is kept only for the equality check between baseline and
    speculative; storing every full continuation would bloat the records file
    without adding evidence.
    """
    record = {
        "kind": kind,
        "prompt_id": prompt_id,
        "k": k,
        "repeat": repeat,
        "total_time_ms": payload["total_time_ms"],
        "tokens_generated": payload["tokens_generated"],
        "target_calls": payload["target_calls"],
        "text": payload["text"],
        "rounds": payload.get("rounds", []),
    }
    for key in ("alpha_hat", "c_hat", "predicted_speedup", "draft_time_ms", "target_time_ms", "gamma"):
        if key in payload:
            record[key] = payload[key]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per cell")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="2 prompts, K in {1,4}, 1 repeat")
    parser.add_argument("--out", default=str(RESULTS_DIR / "records.json"))
    args = parser.parse_args()

    prompts = dict(PROMPTS)
    k_values = list(K_VALUES)
    repeats = args.repeats
    if args.quick:
        prompts = {k: prompts[k] for k in list(prompts)[:2]}
        k_values = [1, 4]
        repeats = 1

    print(f"loading models on {DEVICE} ({DTYPE}) ...", flush=True)
    load_models()

    # One warm-up generation, discarded. The first call pays for lazy CUDA/MPS
    # kernel compilation and allocator warm-up, which would otherwise be
    # charged to whichever cell happened to run first.
    print("warm-up run (discarded) ...", flush=True)
    run_speculative(next(iter(prompts.values())), 4, 16)
    run_baseline(next(iter(prompts.values())), 16)

    records = []
    total_cells = len(prompts) * (1 + len(k_values)) * repeats
    done = 0
    started = time.time()

    for prompt_id, prompt in prompts.items():
        for repeat in range(repeats):
            payload = run_baseline(prompt, args.max_new_tokens)
            records.append(build_record("baseline", prompt_id, prompt, None, repeat, payload))
            done += 1
            print(f"[{done}/{total_cells}] baseline {prompt_id} r{repeat} "
                  f"{payload['total_time_ms']:.0f}ms", flush=True)

            for k in k_values:
                payload = run_speculative(prompt, k, args.max_new_tokens)
                records.append(build_record("speculative", prompt_id, prompt, k, repeat, payload))
                done += 1
                print(f"[{done}/{total_cells}] spec {prompt_id} K={k} r{repeat} "
                      f"{payload['total_time_ms']:.0f}ms "
                      f"alpha={payload.get('alpha_hat')} c={payload.get('c_hat')}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": DEVICE,
            "dtype": str(DTYPE),
            "draft_model": DRAFT_MODEL_NAME,
            "target_model": TARGET_MODEL_NAME,
            "max_new_tokens": args.max_new_tokens,
            "repeats": repeats,
            "k_values": k_values,
            "prompt_ids": list(prompts),
            "prompts": prompts,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "wall_clock_s": round(time.time() - started, 1),
        },
        "records": records,
    }
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"\nwrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
