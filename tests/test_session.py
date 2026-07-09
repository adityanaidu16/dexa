"""Persist-and-resume: lossless serialization + the resume benchmark.

Uses the torch-free FakeBackend — its prefill produces a real KVCache and its
generate is deterministic given the cache, so "resume from loaded state ==
resume from live state" is a faithful test of lossless persistence without a GPU.
"""

from __future__ import annotations

import numpy as np

from dexa.engine.fake import FakeBackend
from dexa.session.state import (
    load_kvcache,
    save_kvcache,
    _f32_to_bf16_bits,
    _bf16_bits_to_f32,
)
from dexa.session.store import SessionStore
from dexa.bench.persist import run_persist_bench
from dexa.core.types import KVCache, LayerKV, ModelSpec


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


def _bf16_kv(seed=0, T=512):
    """A KVCache whose fp32 values are already exact bf16 (as a real bf16 model's
    KV is once upcast to fp32) and whose spec advertises bfloat16."""
    rng = np.random.default_rng(seed)
    spec = ModelSpec(name="m", n_layers=3, n_q_heads=4, n_kv_heads=2, head_dim=8,
                     hidden_size=32, dtype="bfloat16")
    layers = []
    for _ in range(spec.n_layers):
        k = _bf16_bits_to_f32(_f32_to_bf16_bits(
            rng.standard_normal((spec.n_kv_heads, T, spec.head_dim)).astype(np.float32)))
        v = _bf16_bits_to_f32(_f32_to_bf16_bits(
            rng.standard_normal((spec.n_kv_heads, T, spec.head_dim)).astype(np.float32)))
        layers.append(LayerKV(key=k, value=v))
    return KVCache(spec=spec, layers=layers,
                   positions=np.arange(T, dtype=np.int64), token_ids=list(range(T)))


def test_bf16_roundtrip_lossless_and_half_size(tmp_path):
    # a bf16 model: auto precision must be lossless AND ~half the fp32 bytes.
    kv = _bf16_kv()
    p_auto = save_kvcache(kv, tmp_path / "auto.npz")            # follows spec.dtype -> bf16
    p_fp32 = save_kvcache(kv, tmp_path / "fp32.npz", precision="float32")

    kv2 = load_kvcache(p_auto)
    for a, b in zip(kv.layers, kv2.layers):
        assert np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)  # lossless

    # the KV arrays dominate the file; bf16 storage is ~2x smaller than fp32.
    assert p_auto.stat().st_size < 0.6 * p_fp32.stat().st_size


def test_fp16_roundtrip_and_legacy_fp32_default(tmp_path):
    kv = _bf16_kv()
    # values representable in fp16 as well (bf16-of-normal draws are small); fp16
    # storage round-trips through fp32 without changing the decode-relevant bits
    # here we only assert it loads and is close.
    p = save_kvcache(kv, tmp_path / "h.npz", precision="float16")
    kv2 = load_kvcache(p)
    for a, b in zip(kv.layers, kv2.layers):
        assert np.allclose(a.key, b.key, rtol=1e-2, atol=1e-2)

    # a legacy file (fp32 keys, no store_dtype tag) still loads bit-identically.
    pf = save_kvcache(kv, tmp_path / "f.npz", precision="float32")
    z = {k: v for k, v in np.load(pf, allow_pickle=False).items() if k != "store_dtype"}
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **z)
    kv3 = load_kvcache(legacy)
    for a, b in zip(kv.layers, kv3.layers):
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


def test_store_blob_format_and_cross_format_load(tmp_path):
    be = FakeBackend()
    ctx = be.tokenize(" ".join(f"tok{i}" for i in range(40)))
    kv = be.prefill(ctx)
    live = be.generate(kv, [], max_new_tokens=6)

    # blob format: lossless resume + on-disk .dexakv.
    store = SessionStore(tmp_path / "sess", format="blob")
    meta = store.save("s1", kv)
    assert meta["format"] == "blob" and meta["path"].endswith(".dexakv")
    assert store.has("s1") and "s1" in store.list_ids()
    loaded, _ = store.load("s1")
    assert be.generate(loaded, [], max_new_tokens=6) == live

    # an npz store can read a blob a blob-store wrote (load auto-detects by suffix).
    npz_store = SessionStore(tmp_path / "sess", format="npz")
    loaded2, _ = npz_store.load("s1")
    assert be.generate(loaded2, [], max_new_tokens=6) == live

    store.delete("s1")
    assert not store.has("s1")


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
