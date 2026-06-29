"""The cartridge artifact: a portable, trained, compact KV cache for a corpus.

A cartridge is stored as dense per-layer numpy arrays (uniform compact length
``t`` across kv-heads, no attention bias) plus the absolute positions the compact
keys live at and metadata. It converts to/from :class:`~dexa.core.types.CompactCache`
so it serves through the existing backend eval path (``biases = 0``), and it
serializes to a single ``.npz`` file — the unit you ship/version/swap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dexa.core.types import CompactCache, CompactLayer, ModelSpec


@dataclass
class Cartridge:
    """Portable trained compact KV for a corpus.

    Shapes: ``keys``/``values`` are ``[n_layers, n_kv_heads, t, head_dim]``;
    ``positions`` is ``[t]`` (absolute positions the compact keys represent);
    ``logical_length`` is the original corpus length T (so appended query tokens
    get correct RoPE phases).
    """

    spec: ModelSpec
    keys: np.ndarray            # [L, n_kv, t, d] float32
    values: np.ndarray          # [L, n_kv, t, d] float32
    positions: np.ndarray       # [t] int64
    logical_length: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def t(self) -> int:
        return int(self.keys.shape[2])

    @property
    def compression_ratio(self) -> float:
        return self.logical_length / self.t if self.t else float("inf")

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)

    # --- interop with the serving/eval path -------------------------------
    def to_compact_cache(self) -> CompactCache:
        """As a :class:`CompactCache` (biases=0) usable by any ModelBackend."""
        s = self.spec
        layers: list[CompactLayer] = []
        pos = self.positions.astype(np.int64)
        for li in range(s.n_layers):
            keys = [self.keys[li, h].astype(np.float32) for h in range(s.n_kv_heads)]
            values = [self.values[li, h].astype(np.float32) for h in range(s.n_kv_heads)]
            biases = [np.zeros(self.t, dtype=np.float32) for _ in range(s.n_kv_heads)]
            poss = [pos.copy() for _ in range(s.n_kv_heads)]
            layers.append(CompactLayer(keys=keys, values=values, biases=biases, positions=poss))
        return CompactCache(
            spec=s, layers=layers, logical_length=self.logical_length,
            method="cartridge", meta=dict(self.meta),
        )

    @classmethod
    def from_compact_cache(cls, cc: CompactCache, *, meta: dict | None = None) -> "Cartridge":
        """Build from a uniform-budget CompactCache (e.g. an Attention-Matching
        warm start). Requires equal ``t`` across layers/heads."""
        s = cc.spec
        t = cc.layers[0].keys[0].shape[0]
        L, H, d = s.n_layers, s.n_kv_heads, s.head_dim
        keys = np.zeros((L, H, t, d), dtype=np.float32)
        values = np.zeros((L, H, t, d), dtype=np.float32)
        for li, layer in enumerate(cc.layers):
            for h in range(H):
                if layer.keys[h].shape[0] != t:
                    raise ValueError("Cartridge requires uniform compact length t across heads")
                keys[li, h] = layer.keys[h]
                values[li, h] = layer.values[h]
        positions = np.asarray(cc.layers[0].positions[0], dtype=np.int64)
        return cls(spec=s, keys=keys, values=values, positions=positions,
                   logical_length=cc.logical_length, meta={**(cc.meta or {}), **(meta or {})})

    # --- serialization ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        spec_d = {
            "name": self.spec.name, "n_layers": self.spec.n_layers,
            "n_q_heads": self.spec.n_q_heads, "n_kv_heads": self.spec.n_kv_heads,
            "head_dim": self.spec.head_dim, "hidden_size": self.spec.hidden_size,
            "dtype": self.spec.dtype,
        }
        # normalize to a .npz path so the returned path matches what's written
        # (np.savez only auto-appends .npz when the suffix is missing).
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        np.savez_compressed(
            path, keys=self.keys, values=self.values, positions=self.positions,
            logical_length=np.int64(self.logical_length),
            spec=json.dumps(spec_d), meta=json.dumps(self.meta, default=str),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Cartridge":
        z = np.load(Path(path), allow_pickle=False)
        spec_d = json.loads(str(z["spec"]))
        spec = ModelSpec(**spec_d)
        return cls(
            spec=spec,
            keys=z["keys"].astype(np.float32),
            values=z["values"].astype(np.float32),
            positions=z["positions"].astype(np.int64),
            logical_length=int(z["logical_length"]),
            meta=json.loads(str(z["meta"])),
        )
