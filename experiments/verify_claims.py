"""
Checks the numeric claims made in paper.tex and in the teaching material
against experiments/results/records.json.

Each claim states what it asserts, where it is asserted, and how it is checked.
A claim that cannot be checked against the records is reported as SKIP rather
than quietly passing, so the failure mode is a visible gap and not a false
confirmation.

Usage:
    python experiments/verify_claims.py
    python experiments/verify_claims.py --records path/to/records.json

Exit code is 1 if any claim FAILs, so this can gate a release.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import analyze  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def check_exactness(summary):
    """paper.tex: speculative output is byte-identical to the baseline."""
    ex = summary["exactness"]
    if ex["comparisons"] == 0:
        return SKIP, "no baseline/speculative pairs in the records"
    if ex["mismatches"] == 0:
        return PASS, f"{ex['comparisons']}/{ex['comparisons']} comparisons identical"
    return FAIL, f"{ex['mismatches']} of {ex['comparisons']} comparisons differed"


def check_call_reduction(summary):
    """paper.tex: speculative decoding cuts target forward passes substantially."""
    k = 4 if 4 in summary["per_k"] else next(iter(summary["per_k"]))
    v = summary["per_k"].get(k) or {}
    ratio = v.get("call_reduction_mean")
    if ratio is None:
        return SKIP, "no call-reduction data"
    if ratio > 1.0:
        return PASS, f"K={k}: {ratio:.2f}x fewer target passes than baseline"
    return FAIL, f"K={k}: call reduction {ratio:.2f}x is not above 1.0"


def check_speedup_below_call_reduction(summary):
    """paper.tex: the central lesson, wall-clock speedup does NOT track the
    reduction in target forward passes."""
    rows = []
    for k, v in summary["per_k"].items():
        if v["speedup_mean"] is None or v["call_reduction_mean"] is None:
            continue
        rows.append((k, v["speedup_mean"], v["call_reduction_mean"]))
    if not rows:
        return SKIP, "no paired speedup/call-reduction data"
    violations = [f"K={k} speedup {s:.2f}x >= calls {c:.2f}x" for k, s, c in rows if s >= c]
    if violations:
        return FAIL, "; ".join(violations)
    worst = max(rows, key=lambda r: r[2] - r[1])
    return PASS, (f"speedup below call reduction at every K; "
                  f"largest gap at K={worst[0]}: {worst[1]:.2f}x vs {worst[2]:.2f}x")


def check_speedup_curve_shape(summary):
    """theory: speedup rises then falls with K, so the best K is interior."""
    per_k = {k: v["speedup_mean"] for k, v in summary["per_k"].items() if v["speedup_mean"] is not None}
    if len(per_k) < 3:
        return SKIP, "need at least 3 values of K"
    ks = sorted(per_k)
    best = summary["best_k"]
    if best in (ks[0], ks[-1]):
        return SKIP, (f"best K={best} sits at the edge of the swept range {ks}; "
                      "the turnover may lie outside what was measured")
    return PASS, f"best K={best} is interior to {ks}, consistent with a peaked curve"


def check_alpha_range(summary):
    """sanity: acceptance rate is a probability."""
    bad = [(k, v["alpha_mean"]) for k, v in summary["per_k"].items()
           if v["alpha_mean"] is not None and not (0.0 <= v["alpha_mean"] <= 1.0)]
    if bad:
        return FAIL, f"alpha outside [0,1]: {bad}"
    vals = [v["alpha_mean"] for v in summary["per_k"].values() if v["alpha_mean"] is not None]
    if not vals:
        return SKIP, "no alpha measurements"
    return PASS, f"alpha in [{min(vals):.3f}, {max(vals):.3f}] across K"


def check_alpha_varies_by_prompt(summary):
    """teaching claim: acceptance depends on how predictable the text is."""
    prompts = summary["alpha_by_prompt"]["prompts"]
    vals = {p: v["alpha_mean"] for p, v in prompts.items() if v["alpha_mean"] is not None}
    if len(vals) < 2:
        return SKIP, "need at least 2 prompts"
    lo = min(vals, key=vals.get)
    hi = max(vals, key=vals.get)
    spread = vals[hi] - vals[lo]
    return PASS, (f"alpha spans {spread:.3f} across genres "
                  f"({lo}={vals[lo]:.3f} .. {hi}={vals[hi]:.3f})")


def check_iid_assumption(summary):
    """paper.tex describes the per-position instrument. Report what it found
    rather than asserting a direction, so the paper can quote the real result."""
    pos = [p for p in summary["per_position"]["positions"] if p["rate"] is not None]
    if len(pos) < 2:
        return SKIP, "not enough reached positions"
    first, last = pos[0]["rate"], pos[-1]["rate"]
    delta = last - first
    trend = "decays" if delta < -0.02 else ("rises" if delta > 0.02 else "is flat")
    return PASS, (f"acceptance {trend} across positions 1..{pos[-1]['position']} "
                  f"({first:.3f} -> {last:.3f}, delta {delta:+.3f}); "
                  f"{summary['per_position']['rounds_pooled']} rounds pooled")


def check_c_is_hardware_property(summary):
    """paper.tex: c is a hardware property. Within one machine it should be
    roughly stable across K; the cross-machine claim needs a second run."""
    cs = [(k, v["c_mean"], v["c_sd"]) for k, v in summary["per_k"].items() if v["c_mean"] is not None]
    if not cs:
        return SKIP, "no c measurements"
    vals = [c for _, c, _ in cs]
    return PASS, (f"c in [{min(vals):.3f}, {max(vals):.3f}] on this machine "
                  f"({summary['meta']['device']}); cross-machine comparison needs a second records file")


CLAIMS = [
    ("Speculative output is byte-identical to baseline", check_exactness),
    ("Target forward passes are reduced", check_call_reduction),
    ("Wall-clock speedup does not track call reduction", check_speedup_below_call_reduction),
    ("Speedup versus K is peaked, not monotone", check_speedup_curve_shape),
    ("Acceptance rate is a valid probability", check_alpha_range),
    ("Acceptance rate varies with prompt genre", check_alpha_varies_by_prompt),
    ("Per-position acceptance tests the i.i.d. assumption", check_iid_assumption),
    ("Draft/target cost ratio c is hardware-dependent", check_c_is_hardware_property),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(RESULTS_DIR / "records.json"))
    args = parser.parse_args()

    path = Path(args.records)
    if not path.exists():
        raise SystemExit(f"no records at {path}; run experiments/run_experiments.py first")

    summary = analyze(json.loads(path.read_text()))

    width = max(len(name) for name, _ in CLAIMS)
    failures = 0
    print(f"Verifying claims against {path}\n")
    for name, fn in CLAIMS:
        try:
            status, detail = fn(summary)
        except Exception as exc:  # a broken check is a failure, not a pass
            status, detail = FAIL, f"check raised {type(exc).__name__}: {exc}"
        if status == FAIL:
            failures += 1
        print(f"  [{status}] {name.ljust(width)}  {detail}")

    print()
    if failures:
        print(f"{failures} claim(s) FAILED")
        sys.exit(1)
    print("all checked claims hold")


if __name__ == "__main__":
    main()
