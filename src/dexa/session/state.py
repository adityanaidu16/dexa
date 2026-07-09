"""Serialize/deserialize a full :class:`KVCache` — the portable session state.

The on-disk form is a single ``.npz``: stacked per-layer keys/values
``[n_layers, n_kv_heads, T, head_dim]`` plus positions, token ids, and the model
spec. Lossless: a loaded cache is bit-identical to the original, so resumed
decode produces identical tokens (this is the correctness guarantee the
benchmark checks).

Storage precision
-----------------
The in-memory :class:`KVCache` is always numpy ``float32`` (the compaction math
and every backend boundary assume it — see :mod:`dexa.core.types`). But a served
model rarely *runs* in fp32: Llama-family serving is ``bfloat16`` on GPU, and the
fp32 numpy is only an upcast artifact at the backend edge. Persisting that fp32
doubles the bytes written and moved on exactly the axis this state object exists
to make cheap — resume latency (memory bandwidth) and portable-state size
(cross-replica / NVMe / object-store move cost).

So by default we persist at the model's **native** precision (``kv.spec.dtype``):
``bfloat16`` stored as its 16-bit truncation, ``float16`` as fp16, ``float32``
unchanged. This is **lossless** whenever the KV genuinely originated in that
precision (a bf16 value upcast to fp32 has its low 16 mantissa bits zero, so the
16-bit form round-trips bit-exact), which is the real serving case. On load the
cache is reconstructed back to fp32, so nothing downstream changes — only the
on-disk / over-the-wire footprint (and the I/O to move it) halves for bf16/fp16
models. Pass ``precision=`` to force a specific storage dtype (e.g. a lossy bf16
of an fp32-native model to trade a little accuracy for half the state); the
per-file ``store_dtype`` recorded in the ``.npz`` makes the choice self-describing
and keeps legacy fp32 files readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexa.core.types import CompactCache, CompactLayer, KVCache, LayerKV, ModelSpec


def kvcache_nbytes(kv: KVCache) -> int:
    return kv.nbytes()


# --- storage-precision helpers (pure numpy) --------------------------------
#: model-dtype string -> compact on-disk storage dtype tag.
_DTYPE_ALIASES = {
    "float32": "float32", "fp32": "float32", "torch.float32": "float32",
    "float16": "float16", "fp16": "float16", "half": "float16", "torch.float16": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16", "torch.bfloat16": "bfloat16",
}


def _resolve_store_dtype(precision: str, spec_dtype: str) -> str:
    """Map a requested ``precision`` (``"auto"`` => follow the model's
    ``spec.dtype``) to a canonical storage tag in {float32, float16, bfloat16}."""
    tag = spec_dtype if precision == "auto" else precision
    key = str(tag).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"unknown storage precision {tag!r}; expected one of "
            "auto/float32/float16/bfloat16 (or a spec.dtype that maps to those)"
        )
    return _DTYPE_ALIASES[key]


def _f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bfloat16 stored as ``uint16`` (round-to-nearest-even).

    numpy has no native bfloat16, so we carry the raw 16 high bits. For a value
    that is already an exact bf16 (the auto/lossless path: a bf16 tensor upcast to
    fp32 has zero low mantissa bits) the rounding is a no-op and the round-trip is
    bit-exact; for a genuine fp32 (opt-in lossy path) it is the nearest bf16."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even: add 0x7FFF + lsb-of-the-kept-bits before truncating.
    # For any finite float u <= 0x7F7FFFFF, so +0x8000 never overflows uint32 —
    # stay in uint32 (avoids an 8-byte-per-element uint64 blow-up on large slabs).
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_f32_to_bf16_bits`: put the 16 bits back in the high half."""
    u32 = np.ascontiguousarray(u16, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _encode(arr: np.ndarray, store_dtype: str) -> np.ndarray:
    """Cast an fp32 array to its compact on-disk representation."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if store_dtype == "float32":
        return a
    if store_dtype == "float16":
        return a.astype(np.float16)
    if store_dtype == "bfloat16":
        return _f32_to_bf16_bits(a)
    raise ValueError(f"unknown store_dtype {store_dtype!r}")


def _decode(arr: np.ndarray, store_dtype: str) -> np.ndarray:
    """Inverse of :func:`_encode`: reconstruct the fp32 in-memory array."""
    if store_dtype == "bfloat16":
        return _bf16_bits_to_f32(arr)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _spec_dict(spec: ModelSpec) -> dict:
    return {
        "name": spec.name, "n_layers": spec.n_layers, "n_q_heads": spec.n_q_heads,
        "n_kv_heads": spec.n_kv_heads, "head_dim": spec.head_dim,
        "hidden_size": spec.hidden_size, "dtype": spec.dtype,
    }


def save_kvcache(
    kv: KVCache, path: str | Path, *, compress: bool = False, precision: str = "auto"
) -> Path:
    """Persist a KVCache to ``path`` (.npz). Returns the written path.

    ``compress=False`` (default) is the realistic serving choice — fast I/O; the
    "instant resume" win is memory bandwidth, not compression. Use compress=True
    only when disk footprint matters more than reload speed.

    ``precision`` picks the on-disk storage dtype (see module docstring):
    ``"auto"`` (default) persists at the model's native precision
    (``kv.spec.dtype``) — lossless for a genuinely bf16/fp16 model and ~2× smaller
    than fp32 — while ``"float32"``/``"float16"``/``"bfloat16"`` force a specific
    dtype. The chosen tag is stored in the file so :func:`load_kvcache`
    reconstructs the fp32 in-memory cache without a hint.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    s = kv.spec
    store_dtype = _resolve_store_dtype(precision, s.dtype)
    keys = np.stack([_encode(l.key, store_dtype) for l in kv.layers])    # [L, n_kv, T, d]
    values = np.stack([_encode(l.value, store_dtype) for l in kv.layers])
    writer = np.savez_compressed if compress else np.savez
    writer(
        path,
        keys=keys, values=values,
        positions=kv.positions.astype(np.int64),
        token_ids=np.asarray(kv.token_ids if kv.token_ids is not None else [], dtype=np.int64),
        spec=json.dumps(_spec_dict(s)),
        meta=json.dumps(kv.meta, default=str),
        store_dtype=str(store_dtype),
    )
    return path


def load_kvcache(path: str | Path) -> KVCache:
    """Load a KVCache previously written by :func:`save_kvcache`.

    Reconstructs the in-memory cache as numpy fp32 regardless of the on-disk
    storage precision (legacy files without a ``store_dtype`` tag are read as
    fp32, preserving backward compatibility)."""
    z = np.load(Path(path), allow_pickle=False)
    spec = ModelSpec(**json.loads(str(z["spec"])))
    store_dtype = str(z["store_dtype"]) if "store_dtype" in z.files else "float32"
    keys = z["keys"]      # [L, n_kv, T, d]
    values = z["values"]
    layers = [
        LayerKV(
            key=np.ascontiguousarray(_decode(keys[li], store_dtype)),
            value=np.ascontiguousarray(_decode(values[li], store_dtype)),
        )
        for li in range(keys.shape[0])
    ]
    tok = z["token_ids"].tolist()
    return KVCache(
        spec=spec, layers=layers, positions=z["positions"].astype(np.int64),
        token_ids=tok if tok else None,
        meta=json.loads(str(z["meta"])),
    )


# --- compacted session state (uniform compact length t across heads) -------
def save_compactcache(cc: CompactCache, path: str | Path, *, compress: bool = False) -> Path:
    """Persist a uniform-budget CompactCache (the *compacted* session state — the
    small object you move at long context). Requires equal ``t`` across heads."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    s = cc.spec
    t = cc.layers[0].keys[0].shape[0]
    L, H, d = s.n_layers, s.n_kv_heads, s.head_dim
    keys = np.zeros((L, H, t, d), dtype=np.float32)
    values = np.zeros((L, H, t, d), dtype=np.float32)
    biases = np.zeros((L, H, t), dtype=np.float32)
    for li, layer in enumerate(cc.layers):
        for h in range(H):
            if layer.keys[h].shape[0] != t:
                raise ValueError("save_compactcache requires uniform compact length t")
            keys[li, h] = layer.keys[h]
            values[li, h] = layer.values[h]
            biases[li, h] = layer.biases[h]
    positions = np.asarray(cc.layers[0].positions[0], dtype=np.int64)
    writer = np.savez_compressed if compress else np.savez
    writer(path, keys=keys, values=values, biases=biases, positions=positions,
           logical_length=np.int64(cc.logical_length),
           spec=json.dumps(_spec_dict(s)), method=str(cc.method),
           meta=json.dumps(cc.meta, default=str))
    return path


def load_compactcache(path: str | Path) -> CompactCache:
    z = np.load(Path(path), allow_pickle=False)
    spec = ModelSpec(**json.loads(str(z["spec"])))
    keys, values, biases = z["keys"], z["values"], z["biases"]
    pos = z["positions"].astype(np.int64)
    layers = []
    for li in range(keys.shape[0]):
        layers.append(CompactLayer(
            keys=[np.ascontiguousarray(keys[li, h]) for h in range(spec.n_kv_heads)],
            values=[np.ascontiguousarray(values[li, h]) for h in range(spec.n_kv_heads)],
            biases=[np.ascontiguousarray(biases[li, h]) for h in range(spec.n_kv_heads)],
            positions=[pos.copy() for _ in range(spec.n_kv_heads)],
        ))
    return CompactCache(spec=spec, layers=layers, logical_length=int(z["logical_length"]),
                        method=str(z["method"]), meta=json.loads(str(z["meta"])))
