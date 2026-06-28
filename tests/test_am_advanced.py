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


# --- ill-conditioned value fit (dead compact keys) ------------------------
def test_value_fit_stays_finite_with_dead_keys():
    """Repeated tokens make several kept keys collinear; the bias fit zeroes all
    but one (``beta = log(1e-9)``), so those keys get ~0 attention weight across
    every reference query -- a near-zero column of the value-fit matrix ``X``.

    Regression: a plain lstsq/normal-equations solve sent those unconstrained
    ``Cv`` rows to ~1e10 (float32 overflow). The fit must drop / bound them so
    the compact values stay finite and on the scale of the original ``V``.
    """
    be = FakeBackend(n_layers=1, n_q_heads=2, n_kv_heads=1, head_dim=8)
    rng = np.random.default_rng(0)
    # Heavy repetition -> many duplicate (collinear) keys.
    token_ids = [7] * 40 + [9] * 40 + [int(x) for x in rng.integers(0, 4, size=40)]
    cache = be.prefill(token_ids)
    ref = be.reference_queries([7, 9, 1, 2, 3])

    # Budget big enough that key selection keeps several of the duplicates.
    am = AttentionMatching().compact(
        cache, CompactionBudget(tokens_per_head=20), ref_queries=ref
    )

    v_absmax = max(float(np.abs(l.value).max()) for l in cache.layers)
    for cl in am.layers:
        for cv in cl.values:
            assert np.isfinite(cv).all(), "compact values must be finite"
            # No value may escape ~the magnitude of the V it stands in for. The
            # pre-fix overflow was ~10 orders larger (~1e10 vs ~1), so a 10x
            # bound is a strict, hash-seed-robust regression guard.
            assert float(np.abs(cv).max()) <= 10.0 * v_absmax


def test_value_fit_no_overflow_warnings():
    """Compaction over a duplicate-heavy context must not raise float overflow
    warnings from the value fit."""
    be = FakeBackend(n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8)
    rng = np.random.default_rng(1)
    token_ids = [3] * 60 + [11] * 60 + [int(x) for x in rng.integers(0, 6, size=60)]
    cache = be.prefill(token_ids)
    ref = be.reference_queries([3, 11, 0, 1, 2, 4])

    old = np.seterr(over="raise", invalid="raise")
    try:
        am = AttentionMatching().compact(
            cache, CompactionBudget(ratio=6.0), ref_queries=ref
        )
    finally:
        np.seterr(**old)
    assert isinstance(am, CompactCache)
    assert all(np.isfinite(v).all() for l in am.layers for v in l.values)


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


def test_hybrid_selection_keeps_high_mass_keys():
    """Mass-aware selection must keep top-attention-mass keys (H2O criterion)
    that pure-importance selection can drop — the long-context needle fix.
    Also checks the recent-window and the dedup/backfill to exactly t keys."""
    import numpy as np
    from dexa.compaction.attention_matching import AttentionMatching, _softmax

    rng = np.random.default_rng(0)
    full_logits = rng.standard_normal((8, 12)).astype(np.float32)
    A = _softmax(full_logits, axis=-1)
    mass = A.sum(axis=0)
    importance = np.sqrt((A ** 2).mean(axis=0))
    t, T = 4, 12

    # mass_frac=1.0 -> exactly the top-t by mass
    am_mass = AttentionMatching(mass_frac=1.0)
    S = am_mass._select_hybrid(full_logits, t, T)
    assert len(S) == t and len(set(S)) == t
    assert set(S.tolist()) == set(np.argsort(mass)[::-1][:t].tolist())

    # recent_frac=1.0 -> exactly the last t positions
    am_recent = AttentionMatching(recent_frac=1.0)
    Sr = am_recent._select_hybrid(full_logits, t, T)
    assert set(Sr.tolist()) == set(range(T - t, T))

    # hybrid: a recent slot + a mass slot + importance backfill, all distinct, len t
    am_hy = AttentionMatching(mass_frac=0.5, recent_frac=0.25)
    Sh = am_hy._select_hybrid(full_logits, t, T)
    assert len(Sh) == t and len(set(Sh.tolist())) == t
    assert (T - 1) in set(Sh.tolist())  # the most-recent position is reserved
    top_mass = set(np.argsort(mass)[::-1][:2].tolist())
    assert top_mass & set(Sh.tolist())  # at least one top-mass key kept
