"""Tests for the bounded iterative working memory (FakeBackend only, no torch).

These exercise the memory-management contract that makes the long-horizon story
work: the maintained KV stays under budget, compaction actually fires on a long
trajectory, decode/score runs against the fused cache, and a snapshot is an
independent copy (the versioning primitive).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from dexa.bench.agentic import make_trajectory, run_agentic
from dexa.compaction.baselines import build as build_compactor
from dexa.engine.fake import FakeBackend
from dexa.memory import WorkingMemory


@pytest.fixture(scope="module")
def backend():
    return FakeBackend(n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8)


def _build_memory(backend, compactor_name, budget=40, keep_recent=16):
    comp = build_compactor(compactor_name)
    return WorkingMemory(
        backend, comp, budget_tokens=budget, keep_recent_tokens=keep_recent, ref_per_head=32
    )


@pytest.mark.parametrize("compactor_name", ["attention_matching", "heavy_hitter", "recent_window"])
def test_memory_stays_under_budget_and_compacts(backend, compactor_name):
    wm = _build_memory(backend, compactor_name, budget=40, keep_recent=16)
    rng = np.random.default_rng(0)
    # ~8 turns of 15 tokens => 120 tokens, well over the 40-token budget.
    for _ in range(8):
        chunk = [int(x) for x in rng.integers(0, 200, size=15)]
        wm.append(chunk)
        # the maintained working set is bounded after every append+compaction
        assert wm.current_tokens <= wm.budget_tokens

    st = wm.stats()
    assert st["n_compactions"] > 0
    assert st["current_tokens"] <= 40
    assert st["total_logical_tokens"] == 120


def test_query_runs_against_fused_cache(backend):
    wm = _build_memory(backend, "attention_matching")
    rng = np.random.default_rng(1)
    for _ in range(6):
        wm.append([int(x) for x in rng.integers(0, 200, size=15)])
    assert wm.n_compactions > 0  # ensures the fused (compact ; recent) path

    prompt = [int(x) for x in rng.integers(0, 200, size=4)]
    gen = wm.query(prompt, max_new_tokens=3)
    assert isinstance(gen, list) and len(gen) == 3

    scores = wm.query(prompt, score_targets=[5, 6])
    assert scores.shape == (2,)


def test_snapshot_is_independent_copy(backend):
    wm = _build_memory(backend, "heavy_hitter")
    rng = np.random.default_rng(2)
    for _ in range(5):
        wm.append([int(x) for x in rng.integers(0, 200, size=15)])
    assert wm.compact_mem is not None

    snap = wm.snapshot()
    snap_keys_before = copy.deepcopy(snap.compact.layers[0].keys[0])

    # keep mutating the live memory; the snapshot must not change.
    for _ in range(3):
        wm.append([int(x) for x in rng.integers(0, 200, size=15)])
    # mutate the live compact memory in place as well
    wm.compact_mem.layers[0].keys[0][:] += 1.0

    assert np.array_equal(snap.compact.layers[0].keys[0], snap_keys_before)
    assert snap.total < wm.total


def test_commit_bumps_version(backend):
    wm = _build_memory(backend, "recent_window")
    rng = np.random.default_rng(3)
    for _ in range(4):
        wm.append([int(x) for x in rng.integers(0, 200, size=15)])
    v0 = wm.snapshot().version
    h = wm.commit()
    assert h.version == v0 + 1


def test_run_agentic_smoke(backend, tmp_path):
    out = tmp_path / "agentic.json"
    res = run_agentic(
        backend,
        strategies=["full_kv", "truncate_recent", "dexa:attention_matching", "dexa:heavy_hitter"],
        n_turns=6,
        turn_tokens=20,
        budget_tokens=40,
        keep_recent_tokens=16,
        n_facts=3,
        seeds=1,
        out_path=str(out),
        verbose=False,
    )
    assert out.exists()
    strategies = {s["strategy"] for s in res["summary"]}
    assert "dexa:attention_matching" in strategies
    # dexa + truncate are bounded; full_kv peak should exceed the budget.
    by = {s["strategy"]: s for s in res["summary"]}
    assert by["truncate_recent"]["peak_tokens"] <= 40
    # dexa's maintained peak is bounded well below the unbounded full-KV cache
    # (transient peak may briefly exceed the budget by up to one turn's tokens).
    assert by["dexa:attention_matching"]["peak_tokens"] < by["full_kv"]["peak_tokens"]
    assert by["full_kv"]["peak_tokens"] > 40
    assert by["dexa:attention_matching"]["n_compactions"] > 0
