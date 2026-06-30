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

from dexa.core.types import KVCache, LayerKV, ModelSpec


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
