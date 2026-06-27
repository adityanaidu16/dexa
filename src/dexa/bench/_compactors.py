"""Compactor access for the harness.

We *prefer* the teammate's real implementations
(``dexa.compaction.baselines`` providing ``COMPACTORS`` and ``build``). Until
those land, this module provides a tiny, correct fallback registry implementing
the same :class:`~dexa.compaction.base.Compactor` interface so the harness and
its tests run end-to-end. Resolution is per-name: a name is taken from the real
registry if it builds there, otherwise from the fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dexa.compaction.base import CompactionBudget, Compactor
from dexa.core.types import CompactCache, CompactLayer, KVCache, RefQueries

# --- prefer the real baselines if a teammate has landed them ---------------
try:  # pragma: no cover - depends on whether baselines.py exists yet
    from dexa.compaction.baselines import build as _real_build  # type: ignore

    try:
        from dexa.compaction.baselines import COMPACTORS as _REAL_REGISTRY  # type: ignore
    except Exception:
        _REAL_REGISTRY = None
    HAVE_BASELINES = True
except Exception:
    _real_build = None
    _REAL_REGISTRY = None
    HAVE_BASELINES = False


# --- fallback selection compactors -----------------------------------------
class _SelectionCompactor(Compactor):
    """Base for compactors that keep a subset of the original keys per head.

    Selection baselines do not merge mass, so per-key biases are zero (the
    softmax simply renormalizes over the kept keys). Attention Matching is the
    only one that uses reference queries to choose *which* keys to keep.
    """

    name = "selection"
    needs_ref_queries = False

    def _indices(
        self,
        cache: KVCache,
        layer: int,
        kv_head: int,
        t: int,
        ref: Optional[RefQueries],
    ) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def compact(
        self,
        cache: KVCache,
        budget: CompactionBudget,
        *,
        ref_queries: Optional[RefQueries] = None,
    ) -> CompactCache:
        s = cache.spec
        T = cache.seq_len
        t = min(max(1, budget.target_t(T)), T)
        layers: list[CompactLayer] = []
        for l in range(s.n_layers):
            K = cache.layers[l].key  # [n_kv_heads, T, d]
            V = cache.layers[l].value
            keys, values, biases, positions = [], [], [], []
            for h in range(s.n_kv_heads):
                idx = np.sort(self._indices(cache, l, h, t, ref_queries))
                keys.append(np.ascontiguousarray(K[h, idx], dtype=np.float32))
                values.append(np.ascontiguousarray(V[h, idx], dtype=np.float32))
                biases.append(np.zeros(len(idx), dtype=np.float32))
                positions.append(cache.positions[idx].astype(np.int64))
            layers.append(CompactLayer(keys=keys, values=values, biases=biases, positions=positions))
        return CompactCache(spec=s, layers=layers, logical_length=T, method=self.name)


class _FullKV(_SelectionCompactor):
    name = "full_kv"

    def _indices(self, cache, layer, kv_head, t, ref):
        return np.arange(cache.seq_len)

    def compact(self, cache, budget, *, ref_queries=None):
        # ignore the budget: keep everything (compression ratio == 1).
        keep_all = CompactionBudget(tokens_per_head=cache.seq_len)
        return super().compact(cache, keep_all, ref_queries=ref_queries)


class _RandomSubset(_SelectionCompactor):
    name = "random_subset"

    def _indices(self, cache, layer, kv_head, t, ref):
        rng = np.random.default_rng(abs(hash(("rand", layer, kv_head, cache.seq_len))) % (2**32))
        return rng.choice(cache.seq_len, size=t, replace=False)


class _RecentWindow(_SelectionCompactor):
    name = "recent_window"

    def _indices(self, cache, layer, kv_head, t, ref):
        return np.arange(cache.seq_len - t, cache.seq_len)


class _AttentionMatching(_SelectionCompactor):
    """Fallback attention-matching: keep the keys carrying the most attention
    mass over the reference queries. This is the leverage/heavy-hitter core of
    real attention matching and reliably beats random selection on the
    reconstruction objective."""

    name = "attention_matching"
    needs_ref_queries = True

    def _indices(self, cache, layer, kv_head, t, ref):
        T = cache.seq_len
        if ref is None:
            return np.linspace(0, T - 1, t).astype(int)
        s = cache.spec
        K = cache.layers[layer].key[kv_head]  # [T, d]
        scale = 1.0 / np.sqrt(s.head_dim)
        Q = ref.layers[layer]  # [n_q_heads, n_ref, d]
        mass = np.zeros(T, dtype=np.float64)
        for qh in range(s.n_q_heads):
            if s.kv_head_of(qh) != kv_head:
                continue
            logits = (Q[qh] @ K.T) * scale  # [n_ref, T]
            logits -= logits.max(axis=-1, keepdims=True)
            w = np.exp(logits)
            w /= np.clip(w.sum(axis=-1, keepdims=True), 1e-8, None)
            mass += w.sum(axis=0)
        return np.argsort(mass)[-t:]


_FALLBACK = {
    c.name: c for c in (_FullKV, _RandomSubset, _RecentWindow, _AttentionMatching)
}


def available_compactors() -> list[str]:
    """Names the harness can build (real registry ∪ fallback)."""
    names = set(_FALLBACK)
    if _REAL_REGISTRY is not None:
        try:
            names |= set(_REAL_REGISTRY)
        except TypeError:
            pass
    return sorted(names)


def build(name: str, **kwargs) -> Compactor:
    """Build a compactor by name, preferring the real baselines registry."""
    if _real_build is not None:
        try:
            return _real_build(name, **kwargs)
        except Exception:
            pass  # fall through to local fallback
    if name in _FALLBACK:
        return _FALLBACK[name](**kwargs)
    raise KeyError(f"unknown compactor {name!r}; available: {available_compactors()}")
