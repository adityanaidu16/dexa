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
