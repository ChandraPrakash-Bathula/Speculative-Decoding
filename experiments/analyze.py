"""
Aggregates raw records from run_experiments.py into the numbers the paper and
the lesson quote.

Every figure printed here is derived from experiments/results/records.json and
nothing else, so a reader can re-derive any published value from the raw file.

Usage:
    python experiments/analyze.py
    python experiments/analyze.py --records path/to/records.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def per_position_acceptance(rounds, gamma):
    """P(accept position i | the round reached position i).

    Positions after the first rejection are never evaluated by the target, so
    they are excluded from the denominator rather than counted as rejections.
    Treating them as rejections would manufacture a decay that the data does
    not contain.
    """
    reached = [0] * gamma
    accepted = [0] * gamma
    for r in rounds:
        for i in range(min(r["proposed"], gamma)):
            reached[i] += 1
            if i < r["accepted"]:
                accepted[i] += 1
            else:
                break
    return [
        {
            "position": i + 1,
            "reached": reached[i],
            "accepted": accepted[i],
            "rate": (accepted[i] / reached[i]) if reached[i] else None,
        }
        for i in range(gamma)
    ]


def analyze(bundle):
    meta = bundle["meta"]
    records = bundle["records"]

    baselines = defaultdict(list)  # prompt_id -> [record]
    speculatives = defaultdict(list)  # (prompt_id, k) -> [record]
    for rec in records:
        if rec["kind"] == "baseline":
            baselines[rec["prompt_id"]].append(rec)
        else:
            speculatives[(rec["prompt_id"], rec["k"])].append(rec)

    out = {"meta": meta}

    # ---- exactness: speculative output must equal baseline output ----
    checks, mismatches = 0, []
    for (prompt_id, k), recs in speculatives.items():
        base_texts = {b["text"] for b in baselines.get(prompt_id, [])}
        if not base_texts:
            continue
        for rec in recs:
            checks += 1
            if rec["text"] not in base_texts:
                mismatches.append({"prompt_id": prompt_id, "k": k, "repeat": rec["repeat"]})
    out["exactness"] = {
        "comparisons": checks,
        "mismatches": len(mismatches),
        "detail": mismatches[:10],
    }

    # ---- per-K aggregates across prompts and repeats ----
    per_k = {}
    for k in meta["k_values"]:
        speedups, alphas, cs, calls_ratio, predicted = [], [], [], [], []
        for prompt_id in meta["prompt_ids"]:
            base = baselines.get(prompt_id, [])
            spec = speculatives.get((prompt_id, k), [])
            if not base or not spec:
                continue
            base_time = statistics.mean(b["total_time_ms"] for b in base)
            base_calls = statistics.mean(b["target_calls"] for b in base)
            for rec in spec:
                speedups.append(base_time / rec["total_time_ms"])
                calls_ratio.append(base_calls / rec["target_calls"] if rec["target_calls"] else None)
                alphas.append(rec.get("alpha_hat"))
                cs.append(rec.get("c_hat"))
                predicted.append(rec.get("predicted_speedup"))

        s_mean, s_sd = mean_sd(speedups)
        a_mean, a_sd = mean_sd(alphas)
        c_mean, c_sd = mean_sd(cs)
        r_mean, _ = mean_sd([c for c in calls_ratio if c is not None])
        p_mean, _ = mean_sd(predicted)
        per_k[k] = {
            "n": len(speedups),
            "speedup_mean": s_mean,
            "speedup_sd": s_sd,
            "call_reduction_mean": r_mean,
            "alpha_mean": a_mean,
            "alpha_sd": a_sd,
            "c_mean": c_mean,
            "c_sd": c_sd,
            "predicted_speedup_mean": p_mean,
        }
    out["per_k"] = per_k

    # ---- acceptance rate by prompt genre, at the default K ----
    default_k = 4 if 4 in meta["k_values"] else meta["k_values"][0]
    by_prompt = {}
    for prompt_id in meta["prompt_ids"]:
        recs = speculatives.get((prompt_id, default_k), [])
        a_mean, a_sd = mean_sd([r.get("alpha_hat") for r in recs])
        by_prompt[prompt_id] = {"alpha_mean": a_mean, "alpha_sd": a_sd, "n": len(recs)}
    out["alpha_by_prompt"] = {"k": default_k, "prompts": by_prompt}

    # ---- per-position acceptance, pooled over every run at the default K ----
    pooled = []
    for prompt_id in meta["prompt_ids"]:
        for rec in speculatives.get((prompt_id, default_k), []):
            pooled.extend(rec["rounds"])
    out["per_position"] = {
        "k": default_k,
        "rounds_pooled": len(pooled),
        "positions": per_position_acceptance(pooled, default_k),
    }

    # ---- best K by measured speedup ----
    ranked = [(k, v["speedup_mean"]) for k, v in per_k.items() if v["speedup_mean"] is not None]
    out["best_k"] = max(ranked, key=lambda kv: kv[1])[0] if ranked else None
    return out


def render(summary):
    meta = summary["meta"]
    lines = []
    lines.append(f"Device      : {meta['device']} ({meta['dtype']})")
    lines.append(f"Draft/target: {meta['draft_model']} -> {meta['target_model']}")
    lines.append(f"Protocol    : {meta['max_new_tokens']} new tokens, "
                 f"{meta['repeats']} repeats, {len(meta['prompt_ids'])} prompts")
    lines.append("")

    ex = summary["exactness"]
    verdict = "PASS" if ex["mismatches"] == 0 else f"FAIL ({ex['mismatches']} mismatched)"
    lines.append(f"Exactness   : {verdict} over {ex['comparisons']} baseline/speculative comparisons")
    lines.append("")

    lines.append("Speedup and acceptance by speculation length K")
    lines.append(f"{'K':>3} {'speedup':>16} {'calls saved':>12} {'alpha':>14} {'c':>14} {'predicted':>10}")
    for k, v in sorted(summary["per_k"].items()):
        if v["speedup_mean"] is None:
            continue
        lines.append(
            f"{k:>3} "
            f"{v['speedup_mean']:>7.2f}x +/-{v['speedup_sd']:<5.2f} "
            f"{v['call_reduction_mean']:>10.2f}x "
            f"{v['alpha_mean']:>7.3f} +/-{v['alpha_sd']:<4.3f} "
            f"{v['c_mean']:>7.3f} +/-{v['c_sd']:<4.3f} "
            f"{v['predicted_speedup_mean']:>8.2f}x"
        )
    lines.append(f"\nBest measured K: {summary['best_k']}")
    lines.append("")

    ap = summary["alpha_by_prompt"]
    lines.append(f"Acceptance rate by prompt genre (K={ap['k']})")
    for pid, v in ap["prompts"].items():
        if v["alpha_mean"] is None:
            continue
        lines.append(f"  {pid:<14} alpha = {v['alpha_mean']:.3f} +/- {v['alpha_sd']:.3f}  (n={v['n']})")
    lines.append("")

    pp = summary["per_position"]
    lines.append(f"Acceptance by position within a round (K={pp['k']}, {pp['rounds_pooled']} rounds pooled)")
    lines.append("  This is the empirical test of the i.i.d. alpha assumption behind the")
    lines.append("  closed-form speedup. A flat curve supports it; a decaying curve does not.")
    for p in pp["positions"]:
        if p["rate"] is None:
            lines.append(f"  position {p['position']}: never reached")
        else:
            lines.append(f"  position {p['position']}: {p['rate']:.3f}  ({p['accepted']}/{p['reached']} rounds)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(RESULTS_DIR / "records.json"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "summary.json"))
    args = parser.parse_args()

    path = Path(args.records)
    if not path.exists():
        raise SystemExit(f"no records at {path}; run experiments/run_experiments.py first")

    bundle = json.loads(path.read_text())
    summary = analyze(bundle)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    report = render(summary)
    print(report)
    Path(args.out).with_suffix(".txt").write_text(report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
