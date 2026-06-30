"""Serialize/deserialize a full :class:`KVCache` — the portable session state.

The on-disk form is a single ``.npz``: stacked per-layer keys/values
``[n_layers, n_kv_heads, T, head_dim]`` plus positions, token ids, and the model
spec. Lossless: a loaded cache is bit-identical to the original, so resumed
decode produces identical tokens (this is the correctness guarantee the
benchmark checks).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexa.core.types import CompactCache, CompactLayer, KVCache, LayerKV, ModelSpec


def kvcache_nbytes(kv: KVCache) -> int:
    return kv.nbytes()


def _spec_dict(spec: ModelSpec) -> dict:
    return {
        "name": spec.name, "n_layers": spec.n_layers, "n_q_heads": spec.n_q_heads,
        "n_kv_heads": spec.n_kv_heads, "head_dim": spec.head_dim,
        "hidden_size": spec.hidden_size, "dtype": spec.dtype,
    }


def save_kvcache(kv: KVCache, path: str | Path, *, compress: bool = False) -> Path:
    """Persist a KVCache to ``path`` (.npz). Returns the written path.

    ``compress=False`` (default) is the realistic serving choice — fast I/O; the
    "instant resume" win is memory bandwidth, not compression. Use compress=True
    only when disk footprint matters more than reload speed.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    s = kv.spec
    keys = np.stack([l.key.astype(np.float32) for l in kv.layers])    # [L, n_kv, T, d]
    values = np.stack([l.value.astype(np.float32) for l in kv.layers])
    writer = np.savez_compressed if compress else np.savez
    writer(
        path,
        keys=keys, values=values,
        positions=kv.positions.astype(np.int64),
        token_ids=np.asarray(kv.token_ids if kv.token_ids is not None else [], dtype=np.int64),
        spec=json.dumps(_spec_dict(s)),
        meta=json.dumps(kv.meta, default=str),
    )
    return path


def load_kvcache(path: str | Path) -> KVCache:
    """Load a KVCache previously written by :func:`save_kvcache`."""
    z = np.load(Path(path), allow_pickle=False)
    spec = ModelSpec(**json.loads(str(z["spec"])))
    keys = z["keys"]      # [L, n_kv, T, d]
    values = z["values"]
    layers = [
        LayerKV(key=np.ascontiguousarray(keys[li]), value=np.ascontiguousarray(values[li]))
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
