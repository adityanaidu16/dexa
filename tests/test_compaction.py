"""Tests for the compaction-algorithms module.

The quality signal is **attention-output reconstruction error** on a HELD-OUT
set of reference queries: caches are compacted using query set A and evaluated
against the full-cache output on a disjoint query set B. This guards against the
classic bug of evaluating on the same queries the bias/value fit saw.
"""

from __future__ import annotations

import numpy as np
import pytest

from dexa.compaction.attention_matching import AttentionMatching
from dexa.compaction.baselines import (
    COMPACTORS,
    FullKV,
    HeavyHitter,
    RandomSubset,
    RecentWindow,
    SnapKVLite,
    build,
)
from dexa.compaction.base import CompactionBudget
from dexa.core.types import CompactCache, RefQueries
from dexa.engine.fake import FakeBackend


# --- fixtures / helpers ---------------------------------------------------
@pytest.fixture(scope="module")
def backend() -> FakeBackend:
    return FakeBackend(n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8)


@pytest.fixture(scope="module")
def context(backend):
    # ~200 token context with a planted vocabulary; deterministic.
    rng = np.random.default_rng(123)
    token_ids = [int(x) for x in rng.integers(0, 300, size=200)]
    cache = backend.prefill(token_ids)
    return token_ids, cache


def _split_ref_queries(backend, token_ids):
    """Two disjoint reference-query sets (train A, eval B) from disjoint token
    populations, so the eval queries are genuinely held out."""
    rng = np.random.default_rng(777)
    toks_a = [int(x) for x in rng.integers(0, 300, size=120)]
    toks_b = [int(x) for x in rng.integers(300, 600, size=120)]
    ref_a = backend.reference_queries(toks_a)
    ref_b = backend.reference_queries(toks_b)
    return ref_a, ref_b


def _recon_error(backend, full_cache, compact_cache, ref_eval):
    """Mean cosine distance and relative L2 over all q-heads/queries/layers."""
    full_out = backend.attention_outputs(full_cache, ref_eval)
    comp_out = backend.attention_outputs(compact_cache, ref_eval)

    cos_dists = []
    num = 0.0
    den = 0.0
    for fo, co in zip(full_out, comp_out):
        f = fo.reshape(-1, fo.shape[-1])
        c = co.reshape(-1, co.shape[-1])
        fn = f / np.clip(np.linalg.norm(f, axis=-1, keepdims=True), 1e-12, None)
        cn = c / np.clip(np.linalg.norm(c, axis=-1, keepdims=True), 1e-12, None)
        cos_dists.append(1.0 - np.sum(fn * cn, axis=-1))
        num += float(np.sum((f - c) ** 2))
        den += float(np.sum(f ** 2))
    cos = float(np.mean(np.concatenate(cos_dists)))
    rel_l2 = float(np.sqrt(num / max(den, 1e-12)))
    return cos, rel_l2


# --- tests ----------------------------------------------------------------
@pytest.mark.parametrize("ratio", [4.0, 10.0])
def test_attention_matching_beats_baselines(backend, context, ratio):
    token_ids, cache = context
    ref_a, ref_b = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=ratio)

    am = AttentionMatching().compact(cache, budget, ref_queries=ref_a)
    rnd = RandomSubset().compact(cache, budget)
    rec = RecentWindow().compact(cache, budget)

    am_cos, am_l2 = _recon_error(backend, cache, am, ref_b)
    rnd_cos, rnd_l2 = _recon_error(backend, cache, rnd, ref_b)
    rec_cos, rec_l2 = _recon_error(backend, cache, rec, ref_b)

    print(
        f"\n[ratio {ratio}x] held-out recon error (cosine / relL2):"
        f"\n  AttentionMatching: {am_cos:.5f} / {am_l2:.5f}"
        f"\n  RandomSubset:      {rnd_cos:.5f} / {rnd_l2:.5f}"
        f"\n  RecentWindow:      {rec_cos:.5f} / {rec_l2:.5f}"
    )

    assert am_cos < rnd_cos, "AttentionMatching should beat RandomSubset (cosine)"
    assert am_l2 < rnd_l2, "AttentionMatching should beat RandomSubset (relL2)"
    assert am_cos < rec_cos, "AttentionMatching should beat RecentWindow (cosine)"
    assert am_l2 < rec_l2, "AttentionMatching should beat RecentWindow (relL2)"


def test_full_kv_is_near_zero_error(backend, context):
    token_ids, cache = context
    _, ref_b = _split_ref_queries(backend, token_ids)
    full = FullKV().compact(cache, CompactionBudget(ratio=1.0))
    cos, l2 = _recon_error(backend, cache, full, ref_b)
    assert cos < 1e-6
    assert l2 < 1e-5
    assert abs(full.compression_ratio - 1.0) < 1e-6


def test_compact_cache_shapes_and_types(backend, context):
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=10.0)
    t = budget.target_t(cache.seq_len)

    am = AttentionMatching().compact(cache, budget, ref_queries=ref_a)
    assert isinstance(am, CompactCache)
    assert am.logical_length == cache.seq_len
    assert len(am.layers) == backend.spec.n_layers

    for layer in am.layers:
        assert len(layer.keys) == backend.spec.n_kv_heads
        for k, v, b, p in zip(layer.keys, layer.values, layer.biases, layer.positions):
            assert k.shape == (t, backend.spec.head_dim)
            assert v.shape == (t, backend.spec.head_dim)
            assert b.shape == (t,)
            assert p.shape == (t,)
            assert k.dtype == np.float32
            assert v.dtype == np.float32
            assert b.dtype == np.float32

    # compression ratio in the right ballpark (~10x).
    assert 8.0 <= am.compression_ratio <= 12.0


def test_compression_ratio_ballpark_4x(backend, context):
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    am = AttentionMatching().compact(cache, CompactionBudget(ratio=4.0), ref_queries=ref_a)
    assert 3.0 <= am.compression_ratio <= 5.0


def test_registry_and_build(backend, context):
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0)
    expected = {"attention_matching", "full_kv", "recent_window", "heavy_hitter",
                "snapkv", "random_subset"}
    assert expected <= set(COMPACTORS)

    for name in COMPACTORS:
        c = build(name)
        rq = ref_a if c.needs_ref_queries else None
        out = c.compact(cache, budget, ref_queries=rq)
        assert isinstance(out, CompactCache)
        assert out.method == name


def test_attention_matching_beats_h2o_and_snapkv(backend, context):
    """AttentionMatching should also beat the stronger attention-aware
    selection baselines (H2O / SnapKV) since it additionally fits bias+value."""
    token_ids, cache = context
    ref_a, ref_b = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=10.0)

    am = AttentionMatching().compact(cache, budget, ref_queries=ref_a)
    h2o = HeavyHitter().compact(cache, budget, ref_queries=ref_a)
    snap = SnapKVLite().compact(cache, budget, ref_queries=ref_a)

    am_cos, _ = _recon_error(backend, cache, am, ref_b)
    h2o_cos, _ = _recon_error(backend, cache, h2o, ref_b)
    snap_cos, _ = _recon_error(backend, cache, snap, ref_b)
    print(
        f"\n[ratio 10x] AM={am_cos:.5f} H2O={h2o_cos:.5f} SnapKV={snap_cos:.5f}"
    )
    assert am_cos < h2o_cos
    assert am_cos < snap_cos
