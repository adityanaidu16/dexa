"""LMCache-style reuse baseline: the *other* answer to the memory wall.

Two schools of thought fight the long-context cost problem:

* **Reuse + tiering** (LMCache): never recompute a prefix you have already seen,
  and never throw KV away while you can spill it down a tier (GPU -> CPU -> NVMe).
  This *saves compute* (prefix-reuse hits avoid re-prefill) but **does not bound
  memory** -- retained KV grows with the total amount of *unique* context the
  system has ever handled.
* **Compaction** (Dexa): shrink old KV into a small, fixed working set
  (:class:`~dexa.memory.WorkingMemory`). Memory stays flat regardless of how much
  unique context streams through.

This module builds the first axis honestly so the harness can run them head to
head. :class:`LMCacheStrategy` serves a *sequence of requests* over a shared
:class:`~dexa.memory.store.TieredCacheStore`, reusing identical token-block
prefixes (block-granular, prefix-chained hashing, exactly the page-reuse scheme
real KV caches use) and tiering the raw KV, but **never compacting**. It reports
prefix-reuse hit rate, peak retained KV bytes per tier, and the recompute it
avoids (tokens and GPU-seconds via :class:`~dexa.core.types.CostModel`) relative
to a no-cache full-reprefill baseline.

The decisive contrast -- LMCache's retained footprint climbing while a Dexa
working memory stays bounded on the *same* request stream -- is produced by
:func:`compare_with_dexa`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dexa.core.types import CostModel, hash_tokens
from dexa.engine.base import ModelBackend
from dexa.memory.store import TieredCacheStore


def _chunk(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


@dataclass
class RequestResult:
    """Outcome of serving one request through the reuse cache."""

    n_tokens: int
    n_blocks: int
    reused_tokens: int      # loaded from cache, no recompute
    recomputed_tokens: int  # prefilled (prefix miss / evicted)
    reused_blocks: int
    recomputed_blocks: int

    @property
    def reuse_frac(self) -> float:
        return self.reused_tokens / self.n_tokens if self.n_tokens else 0.0


class LMCacheStrategy:
    """Reuse-and-tier KV across requests at token-block granularity, no compaction.

    Parameters
    ----------
    backend:
        Model backend; only :meth:`~dexa.engine.base.ModelBackend.prefill` is used
        (to materialize raw KV for newly-computed blocks).
    store:
        A :class:`~dexa.memory.store.TieredCacheStore`; created with defaults if
        omitted.
    block_size:
        Token-block (page) granularity at which prefixes are hashed and reused.
    tenant:
        Logical namespace for the reuse index.
    cost:
        :class:`~dexa.core.types.CostModel` for translating recompute-avoided
        tokens into GPU-seconds.
    """

    def __init__(
        self,
        backend: ModelBackend,
        store: Optional[TieredCacheStore] = None,
        *,
        block_size: int = 64,
        tenant: str = "default",
        cost: Optional[CostModel] = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.backend = backend
        self.store = store or TieredCacheStore()
        self.block_size = int(block_size)
        self.tenant = tenant
        self.cost = cost or CostModel()

        self.results: list[RequestResult] = []
        # running totals across all served requests
        self.total_tokens = 0
        self.reused_tokens = 0
        self.recomputed_tokens = 0

    # --- serving ----------------------------------------------------------
    def process(self, token_ids: list[int]) -> RequestResult:
        """Serve one request: load every cached prefix block, prefill + store the
        rest. Returns its :class:`RequestResult` and updates running stats."""
        token_ids = list(token_ids)
        blocks = _chunk(token_ids, self.block_size)
        reused_tokens = recomputed_tokens = 0
        reused_blocks = recomputed_blocks = 0
        diverged = False  # once a prefix block misses, all later blocks must recompute
        offset = 0

        for block in blocks:
            end = offset + len(block)
            key = hash_tokens(token_ids[:end])  # prefix-chained: depends on all prior tokens
            handle = self.store.has(key, tenant=self.tenant)
            if not diverged and handle is not None:
                self.store.get(handle)  # load resident KV (models tier latency, promotes)
                reused_tokens += len(block)
                reused_blocks += 1
            else:
                diverged = True
                # prefill this block at its true absolute position; store raw KV.
                kv = self.backend.prefill(block, position_offset=offset)
                self.store.put(key, kv, tenant=self.tenant)
                recomputed_tokens += len(block)
                recomputed_blocks += 1
            offset = end

        res = RequestResult(
            n_tokens=len(token_ids),
            n_blocks=len(blocks),
            reused_tokens=reused_tokens,
            recomputed_tokens=recomputed_tokens,
            reused_blocks=reused_blocks,
            recomputed_blocks=recomputed_blocks,
        )
        self.results.append(res)
        self.total_tokens += res.n_tokens
        self.reused_tokens += reused_tokens
        self.recomputed_tokens += recomputed_tokens
        return res

    # --- reporting --------------------------------------------------------
    def stats(self) -> dict:
        """Aggregate reuse / memory / compute-saving report for the whole run.

        ``full_reprefill_*`` is the no-cache baseline (every request prefilled in
        full); ``recompute_avoided_*`` is what prefix reuse saved against it.
        ``peak_retained_kv_bytes`` / ``peak_bytes_per_tier`` come from the tiered
        store and are the memory-wall signal: they climb with unique context.
        """
        store = self.store.stats()
        full_reprefill_tokens = self.total_tokens
        avoided_tokens = self.reused_tokens
        return {
            "n_requests": len(self.results),
            "total_tokens": self.total_tokens,
            "reused_tokens": self.reused_tokens,
            "recomputed_tokens": self.recomputed_tokens,
            "prefix_reuse_hit_rate": (
                self.reused_tokens / self.total_tokens if self.total_tokens else 0.0
            ),
            # memory wall: retained raw KV never shrinks (no compaction)
            "peak_retained_kv_bytes": store["peak_total_bytes"],
            "peak_bytes_per_tier": store["peak_bytes_per_tier"],
            "current_retained_kv_bytes": store["total_bytes"],
            "n_resident_blocks": store["n_entries"],
            "evictions_dropped": store["evictions_dropped"],
            "demotions": store["demotions"],
            # compute saved by reuse, vs reprefilling every request from scratch
            "full_reprefill_tokens": full_reprefill_tokens,
            "recompute_avoided_tokens": avoided_tokens,
            "full_reprefill_gpu_seconds": float(self.cost.prefill_seconds(full_reprefill_tokens)),
            "recompute_avoided_gpu_seconds": float(self.cost.prefill_seconds(avoided_tokens)),
            "modeled_access_seconds": store["modeled_access_seconds"],
            "store": store,
        }


# --- scenario builders -----------------------------------------------------
def _transcript_stream(
    n_requests: int, turn_tokens: int, seed: int, probe_tokens: int = 8
) -> tuple[list[list[int]], list[list[int]]]:
    """Build aligned ``(turns, requests)``.

    ``turns[k]`` is the novel context added at turn ``k``; ``requests[k]`` is the
    running transcript of turns ``0..k`` followed by a *fresh* probe suffix. So
    ``requests`` shares a growing prefix (the reuse workload) while ``turns`` is
    the clean per-turn append stream a compacted memory consumes -- both expose
    the *same* growing unique context."""
    import random

    rng = random.Random(hash(("lmcache-scenario", n_requests, turn_tokens, seed)) & 0xFFFFFFFF)
    vocab = 50000
    turns: list[list[int]] = []
    requests: list[list[int]] = []
    for _ in range(n_requests):
        turn = [rng.randrange(vocab) for _ in range(turn_tokens)]
        turns.append(turn)
        transcript = [t for tn in turns for t in tn]
        probe = [rng.randrange(vocab) for _ in range(probe_tokens)]  # fresh, never reused
        requests.append(transcript + probe)
    return turns, requests


def shared_prefix_requests(
    backend: ModelBackend,
    n_requests: int = 8,
    turn_tokens: int = 120,
    seed: int = 0,
) -> list[list[int]]:
    """A growing-conversation request stream: request *k* is the running
    transcript of turns ``0..k`` plus a fresh probe suffix.

    This is the canonical reuse workload -- each request re-sends (and so can
    reuse) the entire prior transcript, while still adding new unique tokens.
    Reuse keeps recompute roughly per-turn instead of quadratic, but the *unique*
    context (hence retained KV) grows every turn: the memory wall."""
    return _transcript_stream(n_requests, turn_tokens, seed)[1]


def run_lmcache_scenario(
    backend: ModelBackend,
    requests: Optional[list[list[int]]] = None,
    *,
    block_size: int = 64,
    store: Optional[TieredCacheStore] = None,
    cost: Optional[CostModel] = None,
    n_requests: int = 8,
    turn_tokens: int = 120,
    seed: int = 0,
) -> dict:
    """Serve ``requests`` (or a generated :func:`shared_prefix_requests` stream)
    through an :class:`LMCacheStrategy` and return its :meth:`~LMCacheStrategy.stats`,
    augmented with the per-request reuse trace and the growing-footprint series."""
    if requests is None:
        requests = shared_prefix_requests(
            backend, n_requests=n_requests, turn_tokens=turn_tokens, seed=seed
        )
    strat = LMCacheStrategy(backend, store=store, block_size=block_size, cost=cost)
    footprint_series: list[int] = []
    per_request: list[dict] = []
    for req in requests:
        res = strat.process(req)
        per_request.append(
            {
                "n_tokens": res.n_tokens,
                "reused_tokens": res.reused_tokens,
                "recomputed_tokens": res.recomputed_tokens,
                "reuse_frac": res.reuse_frac,
            }
        )
        footprint_series.append(strat.store.stats()["total_bytes"])

    out = strat.stats()
    out["per_request"] = per_request
    out["retained_bytes_series"] = footprint_series  # monotone-ish: the memory wall
    return out


def compare_with_dexa(
    backend: ModelBackend,
    *,
    n_requests: int = 8,
    turn_tokens: int = 120,
    block_size: int = 64,
    budget_tokens: int = 256,
    keep_recent_tokens: int = 128,
    compactor_name: str = "heavy_hitter",
    seed: int = 0,
    cost: Optional[CostModel] = None,
) -> dict:
    """Head-to-head on one request stream: LMCache (reuse, unbounded memory) vs a
    Dexa :class:`~dexa.memory.WorkingMemory` (compaction, bounded memory).

    The same growing transcript is fed to both. The reuse side's retained KV
    bytes climb turn over turn; the Dexa side's maintained working set stays
    capped at its budget. Returns both footprint series plus headline numbers.
    """
    # local imports to keep this module importable without the compaction stack
    from dexa.bench._compactors import build as build_compactor
    from dexa.memory import WorkingMemory

    cost = cost or CostModel()
    turns, requests = _transcript_stream(n_requests, turn_tokens, seed)

    lm = run_lmcache_scenario(
        backend, requests, block_size=block_size, cost=cost
    )

    # Dexa sees the *same* novel context as a clean turn-by-turn append stream, so
    # logical context grows identically -- but the maintained working set is
    # compacted to a fixed budget after every append.
    wm = WorkingMemory(
        backend,
        build_compactor(compactor_name),
        budget_tokens=budget_tokens,
        keep_recent_tokens=keep_recent_tokens,
        ref_per_head=32,
    )
    kv_bytes_per_tok = 2.0 * backend.spec.n_layers * backend.spec.n_kv_heads * backend.spec.head_dim * 4.0
    dexa_series: list[int] = []
    for turn in turns:
        wm.append(turn)
        # the *maintained* working set after compaction is the bounded footprint
        dexa_series.append(int(wm.current_tokens * kv_bytes_per_tok))

    dexa_stats = wm.stats()
    total_unique = sum(len(t) for t in turns)
    return {
        "lmcache": {
            "peak_retained_kv_bytes": lm["peak_retained_kv_bytes"],
            "retained_bytes_series": lm["retained_bytes_series"],
            "prefix_reuse_hit_rate": lm["prefix_reuse_hit_rate"],
            "recompute_avoided_gpu_seconds": lm["recompute_avoided_gpu_seconds"],
            "bounded": False,
        },
        "dexa": {
            # bounded maintained footprint (post-compaction); the transient
            # re-prefill spike is reported separately, not as retained memory.
            "peak_retained_kv_bytes": int(max(dexa_series) if dexa_series else 0),
            "retained_bytes_series": dexa_series,
            "budget_kv_bytes": int(budget_tokens * kv_bytes_per_tok),
            "n_compactions": dexa_stats["n_compactions"],
            "transient_recompute_tokens": dexa_stats["peak_recompute_tokens"],
            "bounded": True,
        },
        "n_requests": n_requests,
        "total_unique_tokens": total_unique,
    }


DEFAULT_OUT = os.path.join("benchmarks", "out", "lmcache.json")
