"""Tests for the advanced Attention Matching knobs:

* ``budget_alloc="sensitivity"`` -- per-kv-head nonuniform token budgets that
  preserve the per-layer total and honor ``min_tokens``.
* ``value_ridge`` -- ridge-regularized value fit (constructor validation +
  quality preserved on held-out reference queries).
* ``strategy="self_study"`` -- synthetic-continuation reference queries
  (guarded HF smoke test; skipped when torch / a model is unavailable).
"""

from __future__ import annotations

import numpy as np
import pytest

from dexa.compaction.attention_matching import AttentionMatching
from dexa.compaction.baselines import RandomSubset
from dexa.compaction.base import CompactionBudget
from dexa.core.types import CompactCache
from dexa.engine.fake import FakeBackend
from tests.test_compaction import _recon_error, _split_ref_queries


@pytest.fixture(scope="module")
def backend() -> FakeBackend:
    return FakeBackend(n_layers=2, n_q_heads=8, n_kv_heads=4, head_dim=8)


@pytest.fixture(scope="module")
def context(backend):
    rng = np.random.default_rng(2024)
    token_ids = [int(x) for x in rng.integers(0, 300, size=240)]
    return token_ids, backend.prefill(token_ids)


# --- constructor validation ----------------------------------------------
def test_constructor_validation():
    with pytest.raises(ValueError):
        AttentionMatching(budget_alloc="nonsense")
    with pytest.raises(ValueError):
        AttentionMatching(value_ridge=-1.0)
    with pytest.raises(ValueError):
        AttentionMatching(value_train_frac=0.0)
    with pytest.raises(ValueError):
        AttentionMatching(value_train_frac=1.5)


# --- per-head budget allocation ------------------------------------------
def test_sensitivity_preserves_total_budget(backend, context):
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    ratio = 8.0
    budget = CompactionBudget(ratio=ratio)
    t = budget.target_t(cache.seq_len)
    n_kv = backend.spec.n_kv_heads

    am = AttentionMatching(budget_alloc="sensitivity").compact(
        cache, budget, ref_queries=ref_a
    )
    for layer in am.layers:
        per_head = [k.shape[0] for k in layer.keys]
        assert sum(per_head) == t * n_kv          # total preserved
        assert all(b >= 1 for b in per_head)      # min honored
    # the allocation is genuinely nonuniform somewhere
    assert any(
        len(set(k.shape[0] for k in layer.keys)) > 1 for layer in am.layers
    ), "sensitivity allocation should make some head budgets differ"


def test_sensitivity_honors_min_tokens(backend, context):
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0, min_tokens=5)
    am = AttentionMatching(budget_alloc="sensitivity").compact(
        cache, budget, ref_queries=ref_a
    )
    for layer in am.layers:
        assert all(k.shape[0] >= 5 for k in layer.keys)


def test_uniform_is_default_and_unchanged(backend, context):
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0)
    t = budget.target_t(cache.seq_len)
    am = AttentionMatching().compact(cache, budget, ref_queries=ref_a)
    assert am.meta["budget_alloc"] == "uniform"
    for layer in am.layers:
        assert all(k.shape[0] == t for k in layer.keys)


def test_sensitivity_quality_beats_random(backend, context):
    token_ids, cache = context
    ref_a, ref_b = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0)
    am = AttentionMatching(budget_alloc="sensitivity").compact(
        cache, budget, ref_queries=ref_a
    )
    rnd = RandomSubset().compact(cache, budget)
    am_cos, _ = _recon_error(backend, cache, am, ref_b)
    rnd_cos, _ = _recon_error(backend, cache, rnd, ref_b)
    assert am_cos < rnd_cos


# --- ridge value fit ------------------------------------------------------
def test_ridge_runs_and_beats_random(backend, context):
    token_ids, cache = context
    ref_a, ref_b = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0)
    am = AttentionMatching(value_ridge=0.1).compact(cache, budget, ref_queries=ref_a)
    assert isinstance(am, CompactCache)
    assert am.meta["value_ridge"] == 0.1
    rnd = RandomSubset().compact(cache, budget)
    am_cos, _ = _recon_error(backend, cache, am, ref_b)
    rnd_cos, _ = _recon_error(backend, cache, rnd, ref_b)
    assert am_cos < rnd_cos


def test_ridge_changes_values(backend, context):
    """A positive ridge should actually alter the fitted values (vs lstsq)."""
    token_ids, cache = context
    ref_a, _ = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0)
    plain = AttentionMatching(value_ridge=0.0).compact(cache, budget, ref_queries=ref_a)
    ridged = AttentionMatching(value_ridge=0.5).compact(cache, budget, ref_queries=ref_a)
    diffs = [
        float(np.abs(p - r).max())
        for pl, rl in zip(plain.layers, ridged.layers)
        for p, r in zip(pl.values, rl.values)
    ]
    assert max(diffs) > 1e-5


def test_train_frac_runs(backend, context):
    token_ids, cache = context
    ref_a, ref_b = _split_ref_queries(backend, token_ids)
    budget = CompactionBudget(ratio=8.0)
    am = AttentionMatching(value_ridge=0.05, value_train_frac=0.7).compact(
        cache, budget, ref_queries=ref_a
    )
    rnd = RandomSubset().compact(cache, budget)
    am_cos, _ = _recon_error(backend, cache, am, ref_b)
    rnd_cos, _ = _recon_error(backend, cache, rnd, ref_b)
    assert am_cos < rnd_cos


# --- self-study reference queries (HF, guarded) ---------------------------
def test_self_study_reference_queries_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from dexa.engine.hf_backend import HFBackend

        backend = HFBackend()  # tiny-random Llama
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"HF model unavailable: {exc}")

    token_ids = backend.tokenize("the cat sat on the mat near the warm fire") or [1, 2, 3]
    refs = backend.reference_queries(token_ids, strategy="self_study", n_per_head=16)
    s = backend.spec
    assert len(refs.layers) == s.n_layers
    for q in refs.layers:
        assert q.shape[0] == s.n_q_heads
        assert q.shape[2] == s.head_dim
        assert 1 <= q.shape[1] <= 16
        assert np.isfinite(q).all()
