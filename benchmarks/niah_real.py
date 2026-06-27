"""Real-model needle-in-a-haystack demonstration for Dexa compaction.

This is the credibility capstone: it runs a *real* transformer and shows that
Attention Matching preserves the model's ability to recall a planted fact at
high compression, where selection baselines (random / recent / heavy-hitter)
lose it.

Metric — **needle-recall log-probability**. For a context with a planted
"magic number" and the question about it, we teacher-force the gold answer and
sum its per-token log-prob under each cache:

    full-KV          -> the ceiling (no compaction)
    no-context       -> the floor (answer from prior alone)
    <method>@ratio   -> where each compactor lands between them

A compactor that keeps the needle lands near the ceiling; one that drops it
falls toward the floor. This is more faithful for small models than exact-match
generation (a 1B model rarely emits a 7-digit number verbatim, but its answer
distribution still reveals whether the fact survived).

Usage:
    .venv/bin/python benchmarks/niah_real.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --length 800 --ratios 4,8,16 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dexa.bench.tasks import niah_single
from dexa.compaction.base import CompactionBudget
from dexa.compaction.baselines import build as build_compactor


def _sum_logprob(backend, cache, prompt_ids, gold_ids) -> float:
    return float(np.sum(backend.score(cache, prompt_ids, gold_ids)))


def run(model: str, length: int, ratios: list[float], seeds: int, out_dir: Path) -> dict:
    from dexa.engine.hf_backend import HFBackend

    print(f"loading {model} ...", flush=True)
    backend = HFBackend(model)
    spec = backend.spec
    print(f"  spec: {spec.n_layers}L {spec.n_q_heads}q/{spec.n_kv_heads}kv "
          f"head_dim={spec.head_dim}", flush=True)

    methods = ["attention_matching", "heavy_hitter", "snapkv", "recent_window", "random_subset"]
    rows: list[dict] = []

    for seed in range(seeds):
        task = niah_single(backend, length=length, seed=seed)
        ctx, prompt, gold = task.context_ids, task.prompt_ids, task.gold_ids
        T = len(ctx)
        print(f"\n[seed {seed}] key={task.meta['key']} value={task.meta['value']} "
              f"context_tokens={T}", flush=True)

        t0 = time.time()
        full = backend.prefill(ctx)
        refs = backend.reference_queries(ctx, strategy="repeat_prefill", n_per_head=256)
        print(f"  prefill+refs {time.time()-t0:.1f}s", flush=True)

        ceiling = _sum_logprob(backend, full, prompt, gold)
        # no-context floor: score the answer with an empty (1-token) cache
        floor_cache = backend.prefill(ctx[:1])
        floor = _sum_logprob(backend, floor_cache, prompt, gold)
        print(f"  ceiling(full)={ceiling:.2f}  floor(no-ctx)={floor:.2f}", flush=True)
        rows.append({"seed": seed, "method": "full_kv", "ratio": 1.0,
                     "gold_logprob": ceiling, "recall_frac": 1.0, "T": T})
        rows.append({"seed": seed, "method": "no_context", "ratio": float(T),
                     "gold_logprob": floor, "recall_frac": 0.0, "T": T})

        for ratio in ratios:
            budget = CompactionBudget(ratio=ratio)
            for m in methods:
                comp = build_compactor(m)
                tc = time.time()
                kwargs = {"ref_queries": refs} if comp.needs_ref_queries else {}
                cache = comp.compact(full, budget, **kwargs)
                lp = _sum_logprob(backend, cache, prompt, gold)
                # recall fraction: where it lands between floor (0) and ceiling (1)
                rng = (ceiling - floor) or 1e-6
                recall = (lp - floor) / rng
                rows.append({"seed": seed, "method": m, "ratio": float(ratio),
                             "gold_logprob": lp, "recall_frac": recall,
                             "compact_s": time.time() - tc, "T": T})
                print(f"    {m:>18s} @{ratio:>4.0f}x  logprob={lp:8.2f}  "
                      f"recall={recall:5.2f}  ({time.time()-tc:.1f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "niah_real.json").write_text(json.dumps(rows, indent=2))

    # aggregate: mean recall fraction per (method, ratio)
    print("\n=== mean needle-recall fraction (1.0 = full-KV, 0.0 = no context) ===")
    agg: dict = {}
    for r in rows:
        agg.setdefault((r["method"], r["ratio"]), []).append(r["recall_frac"])
    header = f"{'method':>18s} " + "".join(f"{int(rt):>8d}x" for rt in ratios)
    print(header)
    for m in ["attention_matching", "heavy_hitter", "snapkv", "recent_window", "random_subset"]:
        cells = []
        for rt in ratios:
            vals = agg.get((m, float(rt)), [])
            cells.append(f"{np.mean(vals):8.2f}" if vals else f"{'-':>8s}")
        print(f"{m:>18s} " + " ".join(cells))
    print(f"\nraw -> {out_dir / 'niah_real.json'}")
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--length", type=int, default=800, help="filler tokens")
    ap.add_argument("--ratios", default="4,8,16")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-dir", default="benchmarks/out")
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]
    run(args.model, args.length, ratios, args.seeds, Path(args.out_dir))


if __name__ == "__main__":
    main()
