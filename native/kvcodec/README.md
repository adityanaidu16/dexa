# dexa-kvcodec (native accelerator)

Optional **Rust** extension that accelerates the *save* side of Dexa's
memory-mapped KV blob format (`dexa/session/blob.py`).

## What it does — and what it deliberately doesn't

The blob **load** path is already zero-copy (`mmap`): a resumed fp32 KV cache is a
view onto the mapped file, paged in on demand during the host→device copy. There
is nothing for native code to beat there, so **load stays in Python**.

The blob **save** path is the opposite: it must pack every fp32 KV element down to
the model's storage precision (bf16/fp16) and stream the payload to disk. numpy
does that single-threaded and materializes a full intermediate copy
(`.tobytes()`). This crate:

- packs all layers **in parallel** with rayon, releasing the GIL, straight into
  the output byte buffer (no intermediate copy), and
- writes header + payload in one pass.

The bytes it writes are **byte-for-byte identical** to the numpy fallback (bf16 =
round-to-nearest-even on the fp32 bit pattern, matching
`dexa.session.state._f32_to_bf16_bits`; fp16 = IEEE cast; fp32 = raw copy), so
either writer's files load with the same `load_kvcache_blob`. This only changes how
fast the file is produced.

## Status

**Not compiled in the laptop/CI environment** (no Rust toolchain there) — the same
env-gating pattern the repo uses for the vLLM connector. `dexa.session.blob`
imports and runs everywhere on the pure-numpy fallback; when this extension is
built it is picked up automatically. The measured motivation (npz vs mmap blob
load, and the single-threaded numpy pack cost) is in
`benchmarks/persist_format_bench.py`.

## Build

```bash
pip install maturin
maturin develop -m native/kvcodec/pyproject.toml --release
# now: python -c "from dexa.session import kvcodec; print(kvcodec.__doc__)"
```

After that, `SessionStore(..., format="blob").save(...)` (and
`dexa.session.blob.save_kvcache_blob`) use the native pack automatically. Nothing
else changes.
