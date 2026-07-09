"""A memory-mapped binary KV format — the fast persistence path for resume.

Why this exists (measured, not assumed). The default ``.npz`` path
(:mod:`dexa.session.state`) stores the KV inside a ZIP container, so a resume
must parse the archive and **copy every array out** before decode can touch a
single byte — ~68 ms to load a 262 MB slab on this laptop. Resume latency is the
headline metric of Dexa's persistent-state wedge, and that up-front copy is pure
overhead.

This format drops the container: a small JSON header followed by the raw KV
bytes in their storage dtype, 64-byte aligned. Load ``mmap``\\s the file and hands
back **zero-copy numpy views** onto the payload (for the fp32 store dtype), so the
resume path pays only for the pages it actually touches — which happen during the
host→device transfer that has to run anyway. The bf16/fp16 store dtypes still need
a widen-to-fp32 pass on load (numpy has no native bf16), but even then skip the
ZIP parse and land well under npz.

Layout::

    magic   : 8 bytes  b"DEXAKV01"
    hdr_len : 8 bytes  little-endian uint64  (length of the header JSON)
    header  : hdr_len bytes  JSON (spec, store_dtype, shapes, positions,
              token_ids, meta, and the payload byte offset)
    <pad to 64-byte boundary>
    payload : keys bytes || values bytes   (per-layer C-contiguous, storage dtype)

The KV<->bytes precision handling (fp32 / fp16 / bf16-as-uint16) reuses the exact
helpers the ``.npz`` path uses, so the two formats are byte-for-byte equivalent in
*what* they store — this only changes the container and the load mechanics. If the
optional native codec (:mod:`dexa.session.kvcodec`, a Rust extension) is present it
accelerates the *save*-side pack/write; the format on disk is identical either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexa.core.types import KVCache, LayerKV, ModelSpec
from dexa.session.state import (
    _decode,
    _encode,
    _resolve_store_dtype,
    _spec_dict,
)

MAGIC = b"DEXAKV01"
_ALIGN = 64
#: numpy dtype each storage tag occupies on disk.
_STORE_NP = {"float32": np.float32, "float16": np.float16, "bfloat16": np.uint16}


def _try_native():
    """Return the optional Rust codec module if it was built and importable, else
    ``None``. Kept behind a function so importing this module never requires the
    native extension (it is an optional accelerator, not a dependency). Probes both
    the packaged path and the top-level name ``maturin develop`` produces."""
    import importlib
    for name in ("dexa.session.kvcodec", "kvcodec"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def save_kvcache_blob(
    kv: KVCache, path: str | Path, *, precision: str = "auto"
) -> Path:
    """Persist ``kv`` in the mmap binary format. Returns the written path (``.dexakv``).

    ``precision`` matches :func:`dexa.session.state.save_kvcache` (``"auto"`` =>
    the model's native ``spec.dtype``). Uses the native codec for the pack/write
    when available, otherwise a pure-numpy path."""
    path = Path(path)
    if path.suffix != ".dexakv":
        path = path.with_suffix(path.suffix + ".dexakv")
    s = kv.spec
    store_dtype = _resolve_store_dtype(precision, s.dtype)

    n_layers = len(kv.layers)
    # per-layer shape is uniform; capture it once for the header.
    k0 = kv.layers[0].key
    layer_shape = [int(x) for x in k0.shape]  # [n_kv, T, d]
    header = {
        "spec": _spec_dict(s),
        "store_dtype": store_dtype,
        "n_layers": n_layers,
        "layer_shape": layer_shape,
        "positions": kv.positions.astype(np.int64).tolist(),
        "token_ids": list(kv.token_ids) if kv.token_ids is not None else None,
        "meta": kv.meta,
    }
    hdr = json.dumps(header, default=str).encode("utf-8")

    native = _try_native()
    if native is not None:  # pragma: no cover - only when the Rust ext is built
        # Hand the native codec the fp32 layer arrays; it packs to store_dtype and
        # streams header+payload to disk in parallel. Byte-identical to the numpy
        # path below.
        native.save_blob(
            str(path), MAGIC, hdr, store_dtype,
            [np.ascontiguousarray(l.key, dtype=np.float32) for l in kv.layers],
            [np.ascontiguousarray(l.value, dtype=np.float32) for l in kv.layers],
            _ALIGN,
        )
        return path

    # pure-numpy fallback (the tested/running path here). Stream each layer
    # straight to the file — no giant stacked intermediate, no separate tobytes
    # copy (encoded arrays are C-contiguous, so ndarray.tofile writes their raw
    # buffer directly). Keys first, then values, matching the load layout.
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(len(hdr).to_bytes(8, "little"))
        f.write(hdr)
        pad = (-f.tell()) % _ALIGN
        if pad:
            f.write(b"\x00" * pad)
        for l in kv.layers:
            _encode(l.key, store_dtype).tofile(f)
        for l in kv.layers:
            _encode(l.value, store_dtype).tofile(f)
    return path


def load_kvcache_blob(path: str | Path, *, mmap: bool = True, keep_native: bool = False) -> KVCache:
    """Load a KVCache written by :func:`save_kvcache_blob`.

    With ``mmap=True`` (default) the payload is memory-mapped: for the ``float32``
    store dtype the returned layers are **zero-copy views** onto the mapping (the
    resume path pages them in on demand); for ``float16``/``bfloat16`` a widen to
    fp32 is unavoidable (numpy has no native bf16) but the ZIP-parse + full copy of
    the ``.npz`` path is still skipped.

    ``keep_native=True`` skips that widen entirely and hands back the layers **in
    their on-disk store dtype** — ``uint16`` (the raw bf16 bits) for a ``bfloat16``
    store, ``float16`` for ``float16``, ``float32`` unchanged. This keeps the load
    zero-copy for bf16 too (no host-side fp32 pass over the whole slab), which on a
    real 8B/64k resume is the difference between ~25 s and a memcpy — see
    ``docs/RESULTS.md`` (2026-07-09). The returned cache is stamped
    ``meta["native_store_dtype"]`` so a dtype-aware consumer (``HFBackend``) knows a
    ``uint16`` layer is bf16 bits, not integers. **Only pass this to a consumer that
    handles the native representation** — generic fp32 KV math must not touch these
    arrays; use the default (``keep_native=False``) everywhere else."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"{path} is not a DEXAKV blob (magic {magic!r})")
        hdr_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        payload_off = f.tell()
        payload_off += (-payload_off) % _ALIGN

    spec = ModelSpec(**header["spec"])
    store_dtype = header["store_dtype"]
    np_dtype = _STORE_NP[store_dtype]
    n_layers = header["n_layers"]
    n_kv, T, d = header["layer_shape"]
    per_layer = n_kv * T * d
    total = 2 * n_layers * per_layer  # keys + values

    if mmap:
        buf = np.memmap(path, dtype=np_dtype, mode="r", offset=payload_off, shape=(total,))
    else:
        buf = np.fromfile(path, dtype=np_dtype, count=total, offset=payload_off)

    keys = buf[: n_layers * per_layer].reshape(n_layers, n_kv, T, d)
    values = buf[n_layers * per_layer :].reshape(n_layers, n_kv, T, d)

    layers = []
    for li in range(n_layers):
        if keep_native:
            # hand back the raw store-dtype view (uint16 bf16 bits / fp16 / fp32) —
            # no widen, stays zero-copy; the aware consumer reinterprets on device.
            layers.append(LayerKV(key=keys[li], value=values[li]))
        else:
            # _decode is a no-op cast for fp32 (keeps the zero-copy view) and the
            # widen for fp16/bf16.
            layers.append(
                LayerKV(
                    key=_decode(keys[li], store_dtype),
                    value=_decode(values[li], store_dtype),
                )
            )
    tok = header["token_ids"]
    meta = dict(header["meta"])
    if keep_native:
        meta["native_store_dtype"] = store_dtype
    return KVCache(
        spec=spec, layers=layers,
        positions=np.asarray(header["positions"], dtype=np.int64),
        token_ids=list(tok) if tok else None,
        meta=meta,
    )
