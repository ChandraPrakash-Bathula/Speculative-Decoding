# Reproducing the measurements

Every number quoted in the paper and in the lesson comes from this pipeline.
Nothing is hand-copied from a terminal session.

```bash
python experiments/run_experiments.py     # produces results/records.json
python experiments/analyze.py             # produces results/summary.json + summary.txt
python experiments/verify_claims.py       # checks each paper claim, exit 1 on failure
```

A fast smoke run, useful for checking the pipeline works before committing to
the full matrix:

```bash
python experiments/run_experiments.py --quick --max-new-tokens 16 \
    --out experiments/results/smoke.json
python experiments/analyze.py --records experiments/results/smoke.json
```

## What the matrix covers

`run_experiments.py` sweeps 6 prompts across 6 speculation lengths with 3
repeats each, plus a baseline per prompt, for 126 timed generations. The
prompts deliberately span genres (definition, code, explanation, list,
reasoning, creative) because acceptance rate tracks how predictable the
continuation is, and a single prompt would hide that dependence.

One warm-up generation runs first and is discarded. The first call on a device
pays for lazy kernel compilation and allocator warm-up, which would otherwise
be charged to whichever cell happened to run first.

## Read this before quoting a number

**Run it on the hardware you intend to publish.** The draft-to-target cost
ratio `c` is a property of the machine, not of the model pair. On a
memory-constrained laptop the draft model's per-call overhead dominates its
FLOPs and `c` climbs toward the point where speculation stops paying; on a GPU
the same code and the same two models give a much lower `c`. Results from one
machine do not transfer to another, and that fact is itself part of the lesson.

**Close other memory-heavy processes first.** If the machine swaps, the timings
measure the swap, not the models. Note in particular that the FastAPI server
holds its own copy of both models, so stop it before running the harness rather
than paying for two copies.

## Files

| File | Role |
|---|---|
| `run_experiments.py` | runs the matrix, writes raw per-run records, aggregates nothing |
| `analyze.py` | turns raw records into per-K tables, per-position acceptance, and per-genre acceptance |
| `verify_claims.py` | checks each numeric claim against the records; `SKIP` when the records cannot settle it, never a silent pass |
| `results/records.json` | raw measurements plus machine metadata |
| `results/summary.json` | aggregated figures |
| `results/summary.txt` | the same aggregates as a readable report |

The split between recording and aggregating is deliberate: raw records stay
auditable, so any published figure can be traced back to the runs that produced
it, and a disagreement about an aggregate can be settled without re-running the
models.
