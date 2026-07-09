"""Persistence-format microbenchmark: npz (container) vs the mmap blob format.

Resume latency is Dexa's headline metric, and it is dominated by how fast a saved
KV state can be brought back. This isolates that: it builds a realistic KV slab
(no model/GPU — sizes and byte-movement are real) and times save + load for both
formats, plus the state-size win from persisting at the model's native precision.

The load numbers here are the numpy fallback (this laptop has no Rust toolchain);
the native codec (native/kvcodec) only accelerates *save*, and the blob format on
disk is identical with or without it.

  python benchmarks/persist_format_bench.py --layers 32 --kv-heads 8 --tokens 4000 --head-dim 128
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from dexa.core.types import KVCache, LayerKV, ModelSpec
from dexa.session.blob import load_kvcache_blob, save_kvcache_blob
from dexa.session.state import (
    _bf16_bits_to_f32,
    _f32_to_bf16_bits,
    load_kvcache,
    save_kvcache,
)


def _make_kv(L, H, T, d, dtype):
    rng = np.random.default_rng(0)
    spec = ModelSpec(name="bench", n_layers=L, n_q_heads=H, n_kv_heads=H,
                     head_dim=d, hidden_size=H * d, dtype=dtype)
    layers = []
    for _ in range(L):
        k = rng.standard_normal((H, T, d)).astype(np.float32)
        v = rng.standard_normal((H, T, d)).astype(np.float32)
        if dtype == "bfloat16":  # make the fp32 values exact bf16 (real serving case)
            k, v = _bf16_bits_to_f32(_f32_to_bf16_bits(k)), _bf16_bits_to_f32(_f32_to_bf16_bits(v))
        layers.append(LayerKV(key=k, value=v))
    return KVCache(spec=spec, layers=layers, positions=np.arange(T, dtype=np.int64),
                   token_ids=list(range(T)))


def _bench(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


def _touch(kv: KVCache) -> float:
    """Force real access of every KV byte (models the host→device copy a resume
    must do), so a lazy mmap view is charged fairly."""
    return sum(float(l.key.sum()) + float(l.value.sum()) for l in kv.layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    kv = _make_kv(args.layers, args.kv_heads, args.tokens, args.head_dim, args.dtype)
    fp32_mb = kv.nbytes() / 1e6
    print(f"KV slab: {args.layers}L x {args.kv_heads}H x {args.tokens}T x {args.head_dim}d, "
          f"model dtype={args.dtype}  (fp32 in-memory {fp32_mb:.0f} MB)\n")

    tmp = Path(tempfile.mkdtemp())

    # --- npz (current default), fp32-forced vs native-precision ---
    p_npz32 = save_kvcache(kv, tmp / "npz32", precision="float32")
    p_npz = save_kvcache(kv, tmp / "npz", precision="auto")
    npz_save = _bench(lambda: save_kvcache(kv, tmp / "npz", precision="auto"))
    npz_load = _bench(lambda: _touch(load_kvcache(p_npz)))

    # --- mmap blob, native precision ---
    p_blob = save_kvcache_blob(kv, tmp / "blob", precision="auto")
    blob_save = _bench(lambda: save_kvcache_blob(kv, tmp / "blob", precision="auto"))
    blob_load = _bench(lambda: _touch(load_kvcache_blob(p_blob)))

    def mb(p): return p.stat().st_size / 1e6
    print(f"{'format':<22}{'save ms':>10}{'load ms':>10}{'file MB':>10}")
    print(f"{'npz (fp32-forced)':<22}{'-':>10}{'-':>10}{mb(p_npz32):>10.0f}")
    print(f"{'npz (native dtype)':<22}{npz_save*1e3:>10.1f}{npz_load*1e3:>10.1f}{mb(p_npz):>10.0f}")
    print(f"{'blob (native dtype)':<22}{blob_save*1e3:>10.1f}{blob_load*1e3:>10.1f}{mb(p_blob):>10.0f}")

    print("\nwins vs npz-fp32 (the current default persisted an fp32-forced-equivalent):")
    print(f"  state size : {mb(p_npz32)/mb(p_blob):.2f}x smaller  "
          f"(native precision, lossless for {args.dtype})")
    print(f"  resume load: {npz_load/blob_load:.2f}x faster  (mmap blob vs npz container)")
    print("  (save side is further accelerated by the native Rust codec; see native/kvcodec)")


if __name__ == "__main__":
    main()
