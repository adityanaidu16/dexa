"""Persist-and-resume: lossless serialization + the resume benchmark.

Uses the torch-free FakeBackend — its prefill produces a real KVCache and its
generate is deterministic given the cache, so "resume from loaded state ==
resume from live state" is a faithful test of lossless persistence without a GPU.
"""

from __future__ import annotations

import numpy as np

from dexa.engine.fake import FakeBackend
from dexa.session.state import save_kvcache, load_kvcache
from dexa.session.store import SessionStore
from dexa.bench.persist import run_persist_bench


def test_kvcache_roundtrip_bit_identical(tmp_path):
    be = FakeBackend()
    kv = be.prefill(be.tokenize("alpha bravo charlie delta echo foxtrot golf hotel"))
    p = save_kvcache(kv, tmp_path / "s.npz")
    kv2 = load_kvcache(p)
    assert kv2.seq_len == kv.seq_len
    assert kv2.token_ids == kv.token_ids
    assert np.array_equal(kv2.positions, kv.positions)
    for a, b in zip(kv.layers, kv2.layers):
        assert np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)


def test_resume_output_identical_to_live(tmp_path):
    be = FakeBackend()
    ctx = be.tokenize(" ".join(f"tok{i}" for i in range(40)))
    kv = be.prefill(ctx)
    live = be.generate(kv, [], max_new_tokens=6)

    store = SessionStore(tmp_path / "sess")
    store.save("s1", kv)
    assert store.has("s1") and "s1" in store.list_ids()
    loaded, load_s = store.load("s1")
    resumed = be.generate(loaded, [], max_new_tokens=6)

    assert resumed == live, "resumed output must be identical to live (lossless)"
    assert load_s >= 0.0


def test_persist_bench_lossless_and_speedup(tmp_path):
    be = FakeBackend()
    res = run_persist_bench(be, lengths=(64, 256), gen_tokens=4,
                            store=SessionStore(tmp_path / "b"), verbose=False)
    assert len(res["rows"]) == 2
    assert res["summary"]["all_identical"] is True
    for r in res["rows"]:
        assert r["identical_output"] is True
        assert r["state_mb"] > 0 and r["resume_ms"] >= 0


def test_compactcache_roundtrip_and_compaction_persist(tmp_path):
    from dexa.compaction.baselines import build
    from dexa.compaction.base import CompactionBudget
    from dexa.session.state import save_compactcache, load_compactcache
    from dexa.bench.persist import run_compaction_persist_bench

    be = FakeBackend(n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8)
    kv = be.prefill(be.tokenize(" ".join(f"w{i}" for i in range(64))))
    cc = build("recent_window").compact(kv, CompactionBudget(ratio=4.0))
    p = save_compactcache(cc, tmp_path / "cc.npz")
    cc2 = load_compactcache(p)
    assert len(cc2.layers) == len(cc.layers)
    assert np.array_equal(cc2.layers[0].keys[0], cc.layers[0].keys[0])
    assert cc2.logical_length == cc.logical_length

    res = run_compaction_persist_bench(be, length=128, ratios=(4, 16),
                                       store=SessionStore(tmp_path / "cp"), verbose=False)
    rows = {r["ratio"]: r for r in res["rows"]}
    assert rows[1]["size_reduction"] == 1.0
    assert rows[16]["size_reduction"] > rows[4]["size_reduction"] > 1.0  # more compaction -> smaller
