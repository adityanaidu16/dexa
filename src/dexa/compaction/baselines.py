"""Baseline compactors.

These are selection-only methods (each keeps a subset of original keys/values
with ``beta = 0``) used to benchmark Attention Matching apples-to-apples:

- :class:`FullKV`       -- no compaction (upper bound on quality).
- :class:`RecentWindow` -- keep the most recent tokens.
- :class:`HeavyHitter`  -- H2O-style: keep highest accumulated attention mass
  plus a small recent window.
- :class:`SnapKVLite`   -- score keys by attention from only the most recent
  reference queries, plus a recent window.
- :class:`RandomSubset` -- deterministic random subset (the floor baseline).

All share the same :class:`~dexa.compaction.base.Compactor` interface so the
benchmark harness treats them identically. Selection scores aggregate the
query heads that share each kv-head under GQA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dexa.compaction.attention_matching import AttentionMatching, _softmax
from dexa.compaction.base import CompactionBudget, Compactor
from dexa.core.types import (
    CompactCache,
    CompactLayer,
    KVCache,
    RefQueries,
    hash_tokens,
)


def _make_layer(
    K_all: np.ndarray,   # [n_kv_heads, T, d]
    V_all: np.ndarray,   # [n_kv_heads, T, d]
    positions: np.ndarray,
    sel_per_head: list[np.ndarray],
) -> CompactLayer:
    """Build a selection-only CompactLayer (beta=0) from per-head index sets."""
    keys, values, biases, poss = [], [], [], []
    for h, S in enumerate(sel_per_head):
        S = np.sort(S)
        keys.append(K_all[h][S].astype(np.float32))
        values.append(V_all[h][S].astype(np.float32))
        biases.append(np.zeros(S.shape[0], dtype=np.float32))
        poss.append(positions[S].astype(positions.dtype))
    return CompactLayer(keys=keys, values=values, biases=biases, positions=poss)


def _group_queries(Q_all: np.ndarray, spec, h: int) -> np.ndarray:
    """Stack reference queries of every q-head mapping to kv-head ``h``."""
    q_heads = [qh for qh in range(spec.n_q_heads) if spec.kv_head_of(qh) == h]
    return np.concatenate([Q_all[qh] for qh in q_heads], axis=0).astype(np.float32)


def _topk_with_recent(scores: np.ndarray, t: int, T: int, recent: int) -> np.ndarray:
    """Pick ``recent`` most-recent indices plus top-scoring others, total ``t``."""
    recent = min(recent, t, T)
    recent_idx = np.arange(T - recent, T)
    masked = scores.copy()
    masked[recent_idx] = -np.inf
    need = t - recent
    if need > 0:
        extra = np.argsort(masked)[::-1][:need]
        sel = np.concatenate([recent_idx, extra])
    else:
        sel = recent_idx
    return np.unique(sel)


class FullKV(Compactor):
    """No compaction: copy every key/value (t = T, beta = 0)."""

    name = "full_kv"
    needs_ref_queries = False

    def compact(self, cache, budget, *, ref_queries=None) -> CompactCache:
        spec = cache.spec
        layers = []
        for l in range(spec.n_layers):
            sel = [np.arange(cache.seq_len) for _ in range(spec.n_kv_heads)]
            layers.append(
                _make_layer(cache.layers[l].key, cache.layers[l].value, cache.positions, sel)
            )
        return CompactCache(
            spec=spec, layers=layers, logical_length=cache.seq_len, method="full_kv"
        )


class RecentWindow(Compactor):
    """Keep the last ``t`` tokens per head (sliding-window attention)."""

    name = "recent_window"
    needs_ref_queries = False

    def compact(self, cache, budget, *, ref_queries=None) -> CompactCache:
        spec = cache.spec
        T = cache.seq_len
        t = min(budget.target_t(T), T)
        sel = np.arange(T - t, T)
        layers = []
        for l in range(spec.n_layers):
            per_head = [sel for _ in range(spec.n_kv_heads)]
            layers.append(
                _make_layer(cache.layers[l].key, cache.layers[l].value, cache.positions, per_head)
            )
        return CompactCache(spec=spec, layers=layers, logical_length=T, method="recent_window")


class HeavyHitter(Compactor):
    """H2O-style: keep keys with highest accumulated attention mass over the
    reference queries, plus a small recent window (default last 0.2*t)."""

    name = "heavy_hitter"
    needs_ref_queries = True

    def __init__(self, recent_frac: float = 0.2) -> None:
        self.recent_frac = recent_frac

    def compact(self, cache, budget, *, ref_queries=None) -> CompactCache:
        if ref_queries is None:
            raise ValueError("HeavyHitter requires ref_queries")
        spec = cache.spec
        scale = 1.0 / np.sqrt(spec.head_dim)
        T = cache.seq_len
        t = min(budget.target_t(T), T)
        recent = int(round(self.recent_frac * t))

        layers = []
        for l in range(spec.n_layers):
            K_all, V_all = cache.layers[l].key, cache.layers[l].value
            Q_all = ref_queries.layers[l]
            per_head = []
            for h in range(spec.n_kv_heads):
                Q = _group_queries(Q_all, spec, h)
                A = _softmax((Q @ K_all[h].T) * scale, axis=-1)  # [n_ref, T]
                scores = A.sum(axis=0)                            # [T]
                per_head.append(_topk_with_recent(scores, t, T, recent))
            layers.append(_make_layer(K_all, V_all, cache.positions, per_head))
        return CompactCache(spec=spec, layers=layers, logical_length=T, method="heavy_hitter")


class SnapKVLite(Compactor):
    """SnapKV-style: score keys using only the most recent ~min(n_ref, 32)
    reference queries (the "observation window"), plus a recent window."""

    name = "snapkv"
    needs_ref_queries = True

    def __init__(self, window: int = 32, recent_frac: float = 0.2) -> None:
        self.window = window
        self.recent_frac = recent_frac

    def compact(self, cache, budget, *, ref_queries=None) -> CompactCache:
        if ref_queries is None:
            raise ValueError("SnapKVLite requires ref_queries")
        spec = cache.spec
        scale = 1.0 / np.sqrt(spec.head_dim)
        T = cache.seq_len
        t = min(budget.target_t(T), T)
        recent = int(round(self.recent_frac * t))

        layers = []
        for l in range(spec.n_layers):
            K_all, V_all = cache.layers[l].key, cache.layers[l].value
            Q_all = ref_queries.layers[l]
            per_head = []
            for h in range(spec.n_kv_heads):
                Qfull = _group_queries(Q_all, spec, h)
                n_ref = Q_all.shape[1]
                w = min(n_ref, self.window)
                # Last `w` queries of every q-head in the group.
                obs = np.concatenate(
                    [Q_all[qh][n_ref - w:] for qh in range(spec.n_q_heads)
                     if spec.kv_head_of(qh) == h],
                    axis=0,
                ).astype(np.float32)
                A = _softmax((obs @ K_all[h].T) * scale, axis=-1)  # [w*group, T]
                scores = A.sum(axis=0)
                per_head.append(_topk_with_recent(scores, t, T, recent))
            layers.append(_make_layer(K_all, V_all, cache.positions, per_head))
        return CompactCache(spec=spec, layers=layers, logical_length=T, method="snapkv")


class RandomSubset(Compactor):
    """Deterministic random subset of keys (seeded from content). Floor baseline."""

    name = "random_subset"
    needs_ref_queries = False

    def compact(self, cache, budget, *, ref_queries=None) -> CompactCache:
        spec = cache.spec
        T = cache.seq_len
        t = min(budget.target_t(T), T)

        if cache.token_ids is not None:
            seed_hex = hash_tokens(cache.token_ids)
        else:
            seed_hex = hash_tokens([int(p) for p in cache.positions.tolist()])
        seed = int(seed_hex[:8], 16)

        layers = []
        for l in range(spec.n_layers):
            per_head = []
            for h in range(spec.n_kv_heads):
                rng = np.random.default_rng(seed + l * spec.n_kv_heads + h)
                per_head.append(rng.choice(T, size=t, replace=False))
            layers.append(
                _make_layer(cache.layers[l].key, cache.layers[l].value, cache.positions, per_head)
            )
        return CompactCache(spec=spec, layers=layers, logical_length=T, method="random_subset")


# --- registry ------------------------------------------------------------
COMPACTORS: dict[str, type[Compactor]] = {
    AttentionMatching.name: AttentionMatching,
    FullKV.name: FullKV,
    RecentWindow.name: RecentWindow,
    HeavyHitter.name: HeavyHitter,
    SnapKVLite.name: SnapKVLite,
    RandomSubset.name: RandomSubset,
}


def build(name: str, **kwargs) -> Compactor:
    """Instantiate a registered compactor by ``name``."""
    try:
        cls = COMPACTORS[name]
    except KeyError:
        raise ValueError(f"unknown compactor {name!r}; choices: {sorted(COMPACTORS)}") from None
    return cls(**kwargs)
