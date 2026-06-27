"""Tests for the tiered reuse store and the LMCache baseline (FakeBackend, no torch).

These pin the *reuse* axis we contrast Dexa against:

* :class:`~dexa.memory.store.TieredCacheStore` -- protocol conformance, content
  reuse via ``has``, LRU down-tier demotion, and true eviction off the bottom
  tier, plus the modeled per-tier access-latency gradient.
* :class:`~dexa.bench.lmcache_baseline.LMCacheStrategy` -- prefix-block reuse hits
  across requests, and the headline contrast: its retained KV footprint grows
  with unique context while a Dexa :class:`~dexa.memory.WorkingMemory` stays
  bounded on the *same* request stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from dexa.bench.lmcache_baseline import (
    LMCacheStrategy,
    compare_with_dexa,
    run_lmcache_scenario,
    shared_prefix_requests,
)
from dexa.core.types import CacheStore
from dexa.engine.fake import FakeBackend
from dexa.memory.store import TierSpec, TieredCacheStore, default_tiers


@pytest.fixture(scope="module")
def backend():
    return FakeBackend(n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8)


def _kv(backend, tokens, offset=0):
    return backend.prefill(list(tokens), position_offset=offset)


# --- store ----------------------------------------------------------------
def test_store_satisfies_cache_store_protocol():
    store = TieredCacheStore()
    assert isinstance(store, CacheStore)  # runtime_checkable structural match


def test_put_get_has_roundtrip_and_reuse(backend):
    store = TieredCacheStore()
    kv = _kv(backend, [1, 2, 3, 4])
    handle = store.put("k-abc", kv, tenant="t1")

    # content reuse probe finds the resident entry and counts as a hit
    assert store.has("k-abc", tenant="t1") == handle
    assert store.has("k-abc", tenant="other") is None  # tenant-scoped
    assert store.has("nope", tenant="t1") is None

    got = store.get(handle)
    assert got is kv
    assert store.get("missing-handle") is None

    st = store.stats()
    assert st["n_entries"] == 1
    assert st["reuse_hits"] == 1 and st["reuse_lookups"] == 3
    assert st["get_hits"] == 1 and st["get_misses"] == 1


def test_put_dedup_does_not_double_count_bytes(backend):
    store = TieredCacheStore()
    kv = _kv(backend, [5, 6, 7])
    h1 = store.put("dup", kv)
    bytes_after_first = store.stats()["total_bytes"]
    h2 = store.put("dup", kv)  # same (tenant, key) -> dedup
    assert h1 == h2
    assert store.stats()["total_bytes"] == bytes_after_first
    assert store.stats()["n_entries"] == 1


def test_lru_demotion_and_bottom_tier_eviction(backend):
    # one entry per block of tokens; size each so the gpu tier holds ~2 entries.
    kv = _kv(backend, list(range(8)))
    nb = kv.nbytes()
    tiers = [
        TierSpec("gpu", capacity_bytes=2 * nb, read_bandwidth_bytes_per_s=2_000e9),
        TierSpec("cpu", capacity_bytes=2 * nb, read_bandwidth_bytes_per_s=25e9, fixed_latency_s=1e-5),
        TierSpec("nvme", capacity_bytes=1 * nb, read_bandwidth_bytes_per_s=3e9, fixed_latency_s=1e-4),
    ]
    store = TieredCacheStore(tiers)

    handles = []
    for i in range(6):  # total capacity is 5 entries -> 1 must be dropped
        handles.append(store.put(f"e{i}", _kv(backend, list(range(8)))))

    st = store.stats()
    # demotions happened as the gpu/cpu tiers overflowed downward
    assert st["demotions"] > 0
    # capacity is 2+2+1 = 5 entries; the 6th forces one true eviction off nvme
    assert st["evictions_dropped"] >= 1
    assert st["n_entries"] <= 5
    assert st["tiers"]["gpu"]["used_bytes"] <= st["tiers"]["gpu"]["capacity_bytes"]
    # the least-recently-used (oldest) entry is the one dropped
    assert store.has("e0") is None
    assert store.has("e5") is not None


def test_get_promotes_to_top_tier(backend):
    kv = _kv(backend, list(range(8)))
    nb = kv.nbytes()
    tiers = [
        TierSpec("gpu", capacity_bytes=1 * nb, read_bandwidth_bytes_per_s=2_000e9),
        TierSpec("cpu", capacity_bytes=10 * nb, read_bandwidth_bytes_per_s=25e9),
    ]
    store = TieredCacheStore(tiers)
    h0 = store.put("a", _kv(backend, list(range(8))))
    store.put("b", _kv(backend, list(range(8))))  # pushes "a" down to cpu

    # "a" now lives on cpu; getting it promotes it back to gpu
    assert store.get(h0) is not None
    assert store.stats()["promotions"] >= 1


def test_access_latency_gradient(backend):
    kv = _kv(backend, list(range(8)))
    nb = kv.nbytes()
    fast = TieredCacheStore([TierSpec("gpu", 10 * nb, 2_000e9)])
    slow = TieredCacheStore([TierSpec("nvme", 10 * nb, 3e9, fixed_latency_s=1e-4)])
    hf = fast.put("x", _kv(backend, list(range(8))))
    hs = slow.put("x", _kv(backend, list(range(8))))
    fast.get(hf)
    slow.get(hs)
    # the slow (nvme-like) tier models strictly higher access cost
    assert slow.stats()["modeled_access_seconds"] > fast.stats()["modeled_access_seconds"]


def test_evict_is_safe_and_frees_bytes(backend):
    store = TieredCacheStore()
    h = store.put("k", _kv(backend, [1, 2, 3]))
    assert store.stats()["total_bytes"] > 0
    store.evict(h)
    store.evict(h)  # idempotent / safe on unknown handle
    store.evict("never-existed")
    assert store.stats()["total_bytes"] == 0
    assert store.has("k") is None


# --- LMCache strategy ------------------------------------------------------
def test_lmcache_reuses_shared_prefix(backend):
    strat = LMCacheStrategy(backend, block_size=4)
    req = list(range(20))  # 5 blocks of 4

    first = strat.process(req)
    assert first.reused_tokens == 0  # cold: everything recomputed
    assert first.recomputed_tokens == 20

    second = strat.process(req)  # identical request -> full prefix reuse
    assert second.reused_tokens == 20
    assert second.recomputed_tokens == 0

    # extend the request: shared 20-token prefix reused, only the new tail recomputed
    third = strat.process(req + [99, 98, 97, 96])
    assert third.reused_tokens == 20
    assert third.recomputed_tokens == 4

    st = strat.stats()
    assert st["prefix_reuse_hit_rate"] > 0.0
    assert st["recompute_avoided_tokens"] == 40  # 20 + 20
    assert st["recompute_avoided_gpu_seconds"] > 0.0


def test_lmcache_divergent_prefix_forces_recompute(backend):
    strat = LMCacheStrategy(backend, block_size=4)
    strat.process(list(range(12)))
    # share the first block, diverge after -> only the first block reuses
    res = strat.process(list(range(4)) + [100, 101, 102, 103, 104, 105, 106, 107])
    assert res.reused_blocks == 1
    assert res.recomputed_blocks == 2


def test_lmcache_eviction_causes_remiss(backend):
    # a store far too small to retain everything: reuse turns into recompute.
    one = _kv(backend, list(range(4)))
    nb = one.nbytes()
    tiny = TieredCacheStore([TierSpec("gpu", capacity_bytes=nb, read_bandwidth_bytes_per_s=2_000e9)])
    strat = LMCacheStrategy(backend, store=tiny, block_size=4)
    strat.process(list(range(8)))   # 2 blocks, but only 1 fits -> block 0 evicted
    res = strat.process(list(range(8)))
    # the evicted earliest block can no longer be reused
    assert res.reused_tokens < 8
    assert tiny.stats()["evictions_dropped"] >= 1


def test_run_scenario_reports_growing_footprint(backend):
    out = run_lmcache_scenario(
        backend, n_requests=6, turn_tokens=40, block_size=16, seed=1
    )
    series = out["retained_bytes_series"]
    assert len(series) == 6
    # no compaction: retained KV is monotonically non-decreasing and grows
    assert series[-1] > series[0]
    assert np.all(np.diff(series) >= 0)
    assert out["prefix_reuse_hit_rate"] > 0.0
    assert out["peak_retained_kv_bytes"] == max(series)


def test_lmcache_grows_while_dexa_stays_bounded(backend):
    cmp = compare_with_dexa(
        backend,
        n_requests=8,
        turn_tokens=80,
        block_size=32,
        budget_tokens=128,
        keep_recent_tokens=64,
        seed=2,
    )
    lm = cmp["lmcache"]["retained_bytes_series"]
    dx = cmp["dexa"]["retained_bytes_series"]

    # LMCache (reuse, no compaction): footprint climbs with unique context.
    assert lm[-1] > lm[0]
    assert np.all(np.diff(lm) >= 0)

    # Dexa (compaction): maintained working set is bounded by the budget.
    assert cmp["dexa"]["peak_retained_kv_bytes"] <= cmp["dexa"]["budget_kv_bytes"]
    assert max(dx) <= cmp["dexa"]["budget_kv_bytes"]
    assert cmp["dexa"]["n_compactions"] > 0

    # the decisive head-to-head: reuse outgrows the bounded compacted memory.
    assert lm[-1] > max(dx)
    assert cmp["lmcache"]["bounded"] is False and cmp["dexa"]["bounded"] is True


def test_shared_prefix_requests_have_growing_prefix(backend):
    reqs = shared_prefix_requests(backend, n_requests=4, turn_tokens=10, seed=0)
    assert len(reqs) == 4
    # each request's transcript portion extends the previous one (shared prefix)
    for a, b in zip(reqs, reqs[1:]):
        shared = a[:-8]  # drop the fresh 8-token probe suffix
        assert b[: len(shared)] == shared
