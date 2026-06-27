"""Long-horizon agentic workload + memory-strategy evaluation.

This is the "long-context agentic" story end-to-end. A simulated agent runs for
many turns, accumulating filler context with a handful of **planted facts**
("FACT 3: the code for delta is 4821735 .") inserted at early/mid turns. After
the whole trajectory we ask probe questions about facts planted *many turns
earlier* -- a direct test of long-horizon recall.

We replay the *same* trajectory under several memory strategies and measure both
recall and memory:

* ``full_kv``          -- keep everything raw. The recall ceiling, but its KV
  cache grows without bound (the memory wall).
* ``truncate_recent``  -- hard recent-window truncation to the budget. Bounded
  memory, but it forgets anything older than the window.
* ``dexa:<compactor>`` -- :class:`~dexa.memory.WorkingMemory` at the budget,
  iteratively compacting old context while keeping recent context raw. The goal:
  bounded memory *and* high late-recall.

Recall is the **needle-recall log-probability fraction** used by
``benchmarks/niah_real.py``: teacher-force the gold answer under a cache and place
its summed log-prob between a no-context floor (0.0) and the full-KV ceiling
(1.0). A strategy that keeps the fact lands near 1.0; one that drops it falls
toward 0.0.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dexa.bench._compactors import build as _registry_build
from dexa.bench.tasks import _KEYS, _filler, _splice
from dexa.core.types import CostModel
from dexa.memory import WorkingMemory

DEFAULT_OUT = os.path.join("benchmarks", "out", "agentic.json")

# Distinctive single-word "codes" make a stronger, less tokenization-dependent
# recall signal than 7-digit numbers (which a small model rarely reproduces
# verbatim), so the long-horizon effect is measurable even on a 360M model.
_CODEWORDS = [
    "zebra", "mango", "violet", "comet", "harbor", "falcon", "ember", "willow",
    "cobalt", "quartz", "maple", "tundra", "saffron", "orchid", "glacier", "raven",
]
DEFAULT_STRATEGIES = [
    "full_kv",
    "truncate_recent",
    "dexa:attention_matching",
    "dexa:heavy_hitter",
]


# --- trajectory data model -------------------------------------------------
@dataclass
class Probe:
    """A late question about a fact planted earlier in the trajectory."""

    prompt_ids: list[int]
    gold_ids: list[int]
    key: str
    value: str
    plant_turn: int


@dataclass
class Trajectory:
    """A full agent run: per-turn token chunks + probes over early facts."""

    turns: list[list[int]]
    probes: list[Probe]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def all_tokens(self) -> list[int]:
        return [t for turn in self.turns for t in turn]


def make_trajectory(
    backend,
    n_turns: int = 8,
    turn_tokens: int = 150,
    n_facts: int = 4,
    seed: int = 0,
) -> Trajectory:
    """Build a multi-turn trajectory with facts planted in the first half.

    Each turn is ``turn_tokens`` filler tokens; ``n_facts`` distinct facts of the
    form ``"FACT <i>: the code for <key> is <value> ."`` are spliced into early/
    mid turns. Probes ask for each fact, so all probes test recall of context
    that is many turns old by the end of the run.
    """
    rng = random.Random(hash(("agentic", n_turns, turn_tokens, n_facts, seed)) & 0xFFFFFFFF)

    n_facts = min(n_facts, len(_KEYS), len(_CODEWORDS))
    keys = rng.sample(_KEYS, k=n_facts)
    values = rng.sample(_CODEWORDS, k=n_facts)
    # plant facts in the first half of the trajectory so probes are "long horizon"
    plant_horizon = max(1, n_turns // 2)
    plant_turns = sorted(rng.sample(range(plant_horizon), k=min(n_facts, plant_horizon)))
    while len(plant_turns) < n_facts:  # if fewer slots than facts, allow repeats
        plant_turns.append(rng.randrange(plant_horizon))
    plant_turns = sorted(plant_turns)[:n_facts]

    facts_by_turn: dict[int, list[tuple[int, str, str]]] = {}
    for i, (k, v, pt) in enumerate(zip(keys, values, plant_turns)):
        facts_by_turn.setdefault(pt, []).append((i, k, v))

    turns: list[list[int]] = []
    for ti in range(n_turns):
        words = _filler(turn_tokens, rng)
        inserts = []
        for (i, k, v) in facts_by_turn.get(ti, []):
            depth = rng.uniform(0.2, 0.8)
            inserts.append((depth, f"FACT {i}: the code for {k} is {v} ."))
        if inserts:
            words = _splice(words, inserts)
        turns.append(backend.tokenize(" ".join(words)))

    probes: list[Probe] = []
    for i, (k, v, pt) in enumerate(zip(keys, values, plant_turns)):
        question = f"Question: what is the code for {k} ? Answer: the code for {k} is"
        probes.append(
            Probe(
                prompt_ids=backend.tokenize(question),
                gold_ids=backend.tokenize(" " + v),
                key=k,
                value=v,
                plant_turn=pt,
            )
        )

    return Trajectory(
        turns=turns,
        probes=probes,
        meta={
            "n_turns": n_turns,
            "turn_tokens": turn_tokens,
            "n_facts": n_facts,
            "seed": seed,
            "plant_turns": plant_turns,
            "keys": keys,
            "values": values,
        },
    )


# --- scoring helpers -------------------------------------------------------
def _sum_logprob(backend, cache, prompt_ids, gold_ids) -> float:
    return float(np.sum(backend.score(cache, prompt_ids, gold_ids)))


def _kv_bytes_per_token(spec) -> float:
    """Real float32 KV bytes per token for this model (k + v, all layers/heads)."""
    return 2.0 * spec.n_layers * spec.n_kv_heads * spec.head_dim * 4.0


# --- per-strategy runs -----------------------------------------------------
def _run_full_kv(backend, traj: Trajectory) -> dict:
    """Keep everything raw -- the recall ceiling and the unbounded-memory case."""
    all_tokens = traj.all_tokens
    cache = backend.prefill(all_tokens)
    peak = len(all_tokens)
    return {"cache": cache, "peak_tokens": peak, "n_compactions": 0, "compute_seconds": 0.0}


def _run_truncate_recent(backend, traj: Trajectory, budget: int) -> dict:
    """Hard recent-window truncation to ``budget`` tokens (the naive bound)."""
    all_tokens = traj.all_tokens
    kept = all_tokens[-budget:] if budget < len(all_tokens) else all_tokens
    cache = backend.prefill(kept)
    return {
        "cache": cache,
        "peak_tokens": len(kept),
        "n_compactions": 0,
        "compute_seconds": 0.0,
    }


def _run_dexa(
    backend,
    traj: Trajectory,
    compactor_name: str,
    budget: int,
    keep_recent: int,
    ref_strategy: str,
    ref_bank_cap: int,
) -> dict:
    """WorkingMemory with the named compactor: bounded, iteratively compacted."""
    comp = _registry_build(compactor_name)
    wm = WorkingMemory(
        backend,
        comp,
        budget_tokens=budget,
        keep_recent_tokens=keep_recent,
        ref_strategy=ref_strategy,
        ref_per_head=ref_bank_cap,
    )
    for turn in traj.turns:
        wm.append(turn)
    st = wm.stats()
    return {
        "memory": wm,
        "peak_tokens": st["peak_tokens"],
        "n_compactions": st["n_compactions"],
        "compute_seconds": st["compute_seconds"],
        "stats": st,
    }


# --- the evaluator ---------------------------------------------------------
def run_agentic(
    backend,
    strategies: list[str] = DEFAULT_STRATEGIES,
    n_turns: int = 8,
    turn_tokens: int = 150,
    budget_tokens: int = 300,
    *,
    keep_recent_tokens: Optional[int] = None,
    n_facts: int = 4,
    seeds: int = 1,
    ref_strategy: str = "repeat_prefill",
    ref_bank_cap: int = 64,
    cost: Optional[CostModel] = None,
    out_path: Optional[str] = DEFAULT_OUT,
    verbose: bool = True,
) -> dict:
    """Replay the same trajectory under each strategy; score late-recall + memory.

    Returns ``{"rows": [...], "summary": [...]}`` and (optionally) writes it to
    ``out_path``. ``rows`` is per (seed, strategy, probe); ``summary`` aggregates
    mean late-recall / peak tokens / peak KV bytes / compactions per strategy.
    """
    spec = backend.spec
    keep_recent = keep_recent_tokens if keep_recent_tokens is not None else max(1, budget_tokens // 2)
    cost = cost or CostModel(
        name=f"{spec.name}",
        kv_bytes_per_token=_kv_bytes_per_token(spec),
    )

    rows: list[dict] = []
    for seed in range(seeds):
        traj = make_trajectory(
            backend, n_turns=n_turns, turn_tokens=turn_tokens, n_facts=n_facts, seed=seed
        )
        all_tokens = traj.all_tokens
        if verbose:
            print(
                f"\n[seed {seed}] turns={n_turns} turn_tokens={turn_tokens} "
                f"total_tokens={len(all_tokens)} facts={traj.meta['keys']}",
                flush=True,
            )

        # ceiling (full-KV) and floor (no-context) per probe.
        full_cache = backend.prefill(all_tokens)
        floor_cache = backend.prefill(all_tokens[:1])
        ceil = {p.key: _sum_logprob(backend, full_cache, p.prompt_ids, p.gold_ids) for p in traj.probes}
        floor = {p.key: _sum_logprob(backend, floor_cache, p.prompt_ids, p.gold_ids) for p in traj.probes}

        for strat in strategies:
            run = _run_strategy(
                backend, strat, traj, budget_tokens, keep_recent, ref_strategy, ref_bank_cap
            )
            peak_tokens = run["peak_tokens"]
            for p in traj.probes:
                lp = _score_probe(backend, run, p)
                rng = (ceil[p.key] - floor[p.key]) or 1e-6
                recall = (lp - floor[p.key]) / rng
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strat,
                        "key": p.key,
                        "plant_turn": p.plant_turn,
                        "turns_ago": n_turns - p.plant_turn,
                        "gold_logprob": lp,
                        "recall_frac": float(recall),
                        "peak_tokens": int(peak_tokens),
                        "peak_kv_bytes": float(cost.kv_bytes(peak_tokens)),
                        "n_compactions": int(run["n_compactions"]),
                        "compute_seconds": float(run["compute_seconds"]),
                        "total_tokens": len(all_tokens),
                    }
                )
            if verbose:
                rs = [r["recall_frac"] for r in rows if r["seed"] == seed and r["strategy"] == strat]
                print(
                    f"  {strat:>26s}  recall={np.mean(rs):5.2f}  "
                    f"peak_tokens={peak_tokens:5d}  "
                    f"peak_kv={cost.kv_bytes(peak_tokens)/1e6:7.2f}MB  "
                    f"compactions={run['n_compactions']}",
                    flush=True,
                )

    summary = _summarize(rows, cost)
    result = {"rows": rows, "summary": summary, "cost_model": cost.__dict__}
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _run_strategy(backend, strat, traj, budget, keep_recent, ref_strategy, ref_bank_cap) -> dict:
    if strat == "full_kv":
        return _run_full_kv(backend, traj)
    if strat == "truncate_recent":
        return _run_truncate_recent(backend, traj, budget)
    if strat.startswith("dexa:"):
        comp_name = strat.split(":", 1)[1]
        return _run_dexa(
            backend, traj, comp_name, budget, keep_recent, ref_strategy, ref_bank_cap
        )
    raise ValueError(f"unknown strategy {strat!r}")


def _score_probe(backend, run: dict, probe: Probe) -> float:
    if "memory" in run:
        scores = run["memory"].query(probe.prompt_ids, score_targets=probe.gold_ids)
        return float(np.sum(scores))
    return _sum_logprob(backend, run["cache"], probe.prompt_ids, probe.gold_ids)


def _summarize(rows: list[dict], cost: CostModel) -> list[dict]:
    by_strat: dict[str, list[dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r)
    summary = []
    for strat, rs in by_strat.items():
        summary.append(
            {
                "strategy": strat,
                "late_recall": float(np.mean([r["recall_frac"] for r in rs])),
                "peak_tokens": int(max(r["peak_tokens"] for r in rs)),
                "peak_kv_mb": float(max(r["peak_kv_bytes"] for r in rs) / 1e6),
                "n_compactions": int(max(r["n_compactions"] for r in rs)),
                "compute_seconds": float(np.mean([r["compute_seconds"] for r in rs])),
            }
        )
    return summary
