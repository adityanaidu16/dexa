"""Reusable benchmark suites driven by a config (not just ad-hoc scripts).

``run_niah_recall`` is the needle-recall suite (factored from
``benchmarks/niah_real.py`` so the config runner and the script share one code
path). It reports, per (compactor, ratio), the needle-recall fraction
``(lp - floor) / (ceiling - floor)`` where ``ceiling`` is full-KV and ``floor``
is no-context — the metric the real-model results in ``docs/RESULTS.md`` use.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dexa.bench.tasks import niah_single, niah_multikey, multihop
from dexa.compaction.base import CompactionBudget
from dexa.compaction.baselines import build as build_compactor

_TASK_GENERATORS = {
    "niah_single": niah_single,
    "niah_multikey": niah_multikey,
    "multihop": multihop,
}


def _sum_logprob(backend, cache, prompt_ids, gold_ids) -> float:
    return float(np.sum(backend.score(cache, prompt_ids, gold_ids)))


def _make_compactor(name: str, am_opts: dict):
    if name == "attention_matching":
        return build_compactor(
            name,
            budget_alloc=am_opts.get("alloc", "sensitivity"),
            value_ridge=am_opts.get("ridge", 0.05),
            mass_frac=am_opts.get("mass_frac", 0.5),
            recent_frac=am_opts.get("recent_frac", 0.1),
        )
    return build_compactor(name)


def run_niah_recall(
    backend,
    *,
    lengths: list[int] = (800,),
    ratios: list[float] = (8, 16, 32, 64, 128),
    seeds: int = 4,
    task: str = "niah_single",
    ref_strategy: str = "self_study",
    n_ref: int = 256,
    compactors: list[str] = (
        "attention_matching",
        "heavy_hitter",
        "snapkv",
        "recent_window",
        "random_subset",
    ),
    am_opts: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Run the needle-recall suite. Returns ``{"rows": [...], "summary": [...]}``.

    ``rows`` are per (length, seed, method, ratio); ``summary`` aggregates the
    mean recall fraction per (method, ratio, length) plus the paired AM−baseline
    deltas that ``docs/RESULTS.md`` reports.
    """
    am_opts = dict(am_opts or {})
    gen = _TASK_GENERATORS[task]
    rows: list[dict] = []

    for length in lengths:
        for seed in range(seeds):
            t = gen(backend, length=length, seed=seed)
            ctx, prompt, gold = t.context_ids, t.prompt_ids, t.gold_ids
            full = backend.prefill(ctx)
            refs = backend.reference_queries(ctx, strategy=ref_strategy, n_per_head=n_ref)
            ceiling = _sum_logprob(backend, full, prompt, gold)
            floor_cache = backend.prefill(ctx[:1])
            floor = _sum_logprob(backend, floor_cache, prompt, gold)
            rng = (ceiling - floor) or 1e-6
            if verbose:
                print(f"[{task} len={length} seed={seed}] ceiling={ceiling:.2f} "
                      f"floor={floor:.2f} T={len(ctx)}", flush=True)
            rows.append({"length": length, "seed": seed, "method": "full_kv",
                         "ratio": 1.0, "recall_frac": 1.0})
            for ratio in ratios:
                budget = CompactionBudget(ratio=float(ratio))
                for m in compactors:
                    comp = _make_compactor(m, am_opts)
                    kwargs = {"ref_queries": refs} if comp.needs_ref_queries else {}
                    cache = comp.compact(full, budget, **kwargs)
                    lp = _sum_logprob(backend, cache, prompt, gold)
                    recall = (lp - floor) / rng
                    rows.append({"length": length, "seed": seed, "method": m,
                                 "ratio": float(ratio), "gold_logprob": lp,
                                 "recall_frac": recall})
                    if verbose:
                        print(f"  {m:>18s} @{float(ratio):>5.0f}x  recall={recall:5.2f}",
                              flush=True)

    summary = _summarize_recall(rows, ratios)
    return {"rows": rows, "summary": summary}


def _summarize_recall(rows: list[dict], ratios) -> list[dict]:
    """Mean recall per (method, ratio) + paired AM-vs-baseline deltas."""
    methods = sorted({r["method"] for r in rows if r["method"] != "full_kv"})
    out: list[dict] = []
    for ratio in ratios:
        ratio = float(ratio)
        per_method = {}
        for m in methods:
            vals = [r["recall_frac"] for r in rows
                    if r["method"] == m and r["ratio"] == ratio]
            per_method[m] = float(np.mean(vals)) if vals else float("nan")
        row = {"ratio": ratio, **{f"recall:{m}": v for m, v in per_method.items()}}
        # paired delta AM - heavy_hitter (the headline comparison)
        am = [r["recall_frac"] for r in rows
              if r["method"] == "attention_matching" and r["ratio"] == ratio]
        hh = [r["recall_frac"] for r in rows
              if r["method"] == "heavy_hitter" and r["ratio"] == ratio]
        if am and hh and len(am) == len(hh):
            diffs = np.array(am) - np.array(hh)
            row["am_minus_hh_mean"] = float(diffs.mean())
            row["am_wins_frac"] = float((diffs > 0).mean())
        out.append(row)
    return out
