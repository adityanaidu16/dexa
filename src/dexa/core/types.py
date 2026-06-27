"""Core data model for Dexa's compaction-first inference-state engine.

The system is organized around one idea: a model's KV cache for a chunk of
context can be **compacted** — replaced by a much smaller cache (compact keys,
values, and per-key attention biases) that preserves the model's behavior when
the chunk is attended to by future tokens. Dexa owns the lifecycle of that
compact state: producing it, persisting it, versioning it, and stitching it
back into an engine for decode.

Tensor convention
-----------------
Everything at this layer and at the :class:`Compactor` boundary is **numpy
float32**. Torch only appears inside the HF model backend, which converts at its
edges. This keeps the compaction math, the baselines, and the benchmark harness
fully CPU-testable with no GPU/torch dependency.

Shapes (per layer):
    full keys/values   K, V : [n_kv_heads, T, head_dim]   (K is post-RoPE)
    reference queries      Q : [n_q_heads, n_ref, head_dim] (post-RoPE)
    compact keys/values Ck,Cv: [n_kv_heads, t, head_dim]
    compact biases      beta : [n_kv_heads, t]
Under grouped-query attention (GQA) several q-heads map to one kv-head; the
mapping is ``q_head // (n_q_heads // n_kv_heads)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a transformer needed to manipulate its KV cache."""

    name: str
    n_layers: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    dtype: str = "float32"

    @property
    def group_size(self) -> int:
        """Number of query heads sharing each kv head (GQA)."""
        return self.n_q_heads // self.n_kv_heads

    def kv_head_of(self, q_head: int) -> int:
        return q_head // self.group_size


@dataclass
class LayerKV:
    """Full KV for one layer. ``key`` is assumed post-RoPE."""

    key: np.ndarray    # [n_kv_heads, T, head_dim]
    value: np.ndarray  # [n_kv_heads, T, head_dim]

    @property
    def n_tokens(self) -> int:
        return self.key.shape[1]


@dataclass
class KVCache:
    """The full (uncompacted) KV cache for a chunk of context."""

    spec: ModelSpec
    layers: list[LayerKV]
    positions: np.ndarray            # [T] absolute position ids the keys were computed at
    token_ids: Optional[list[int]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return int(self.positions.shape[0])

    def nbytes(self) -> int:
        return sum(l.key.nbytes + l.value.nbytes for l in self.layers)


@dataclass
class CompactLayer:
    """Compact KV for one layer. Budget may differ per kv-head, so entries are
    stored per head as ragged lists."""

    keys: list[np.ndarray]    # per kv_head: [t_h, head_dim]
    values: list[np.ndarray]  # per kv_head: [t_h, head_dim]
    biases: list[np.ndarray]  # per kv_head: [t_h]
    # absolute positions each compact key represents (for RoPE relative phase /
    # logical-length bookkeeping). per kv_head: [t_h]
    positions: list[np.ndarray]

    def budget(self) -> int:
        return sum(k.shape[0] for k in self.keys)


@dataclass
class CompactCache:
    """A compacted KV cache: the persistent, portable Dexa state object.

    ``logical_length`` is the original sequence length T. The physical size is
    ``budget``. New tokens appended after this cache must receive position ids
    starting at ``logical_length`` so RoPE phases stay correct.
    """

    spec: ModelSpec
    layers: list[CompactLayer]
    logical_length: int
    method: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def budget(self) -> int:
        """Total compact tokens summed over layers/heads (physical size proxy)."""
        return sum(layer.budget() for layer in self.layers)

    @property
    def compression_ratio(self) -> float:
        """T / (mean compact tokens per layer per kv-head). Higher = more compression."""
        per_layer = self.spec.n_layers * self.spec.n_kv_heads
        if self.budget == 0:
            return float("inf")
        mean_t = self.budget / per_layer
        return self.logical_length / mean_t if mean_t else float("inf")

    def nbytes(self) -> int:
        return sum(
            sum(k.nbytes for k in l.keys)
            + sum(v.nbytes for v in l.values)
            + sum(b.nbytes for b in l.biases)
            for l in self.layers
        )


@dataclass
class RefQueries:
    """Reference queries used by attention-matching compaction, per layer.

    ``layers[i]`` has shape [n_q_heads, n_ref, head_dim] (post-RoPE).
    """

    spec: ModelSpec
    layers: list[np.ndarray]

    @property
    def n_ref(self) -> int:
        return int(self.layers[0].shape[1]) if self.layers else 0


# --- Cost model -----------------------------------------------------------
@dataclass
class CostModel:
    """Maps token/byte counts to interpretable, hardware-relative units so
    benchmark results read as GPU-seconds / $ / GB, not raw counts. Defaults
    approximate an 8B model on one A100-80G; override per benchmark."""

    name: str = "a100-80g/8B"
    prefill_tok_per_s: float = 9000.0
    decode_tok_per_s: float = 120.0
    gpu_dollars_per_hour: float = 1.80
    kv_bytes_per_token: float = 320_000.0  # ~320KB/token KV for an 8B model

    def prefill_seconds(self, n_tokens: int) -> float:
        return n_tokens / self.prefill_tok_per_s

    def gpu_dollars(self, seconds: float) -> float:
        return seconds * self.gpu_dollars_per_hour / 3600.0

    def kv_bytes(self, n_tokens: int) -> float:
        return n_tokens * self.kv_bytes_per_token


def hash_tokens(token_ids: list[int]) -> str:
    """Stable content hash of a token sequence (identity / dedup of caches)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(len(token_ids).to_bytes(8, "little"))
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=True))
    return h.hexdigest()


@runtime_checkable
class CacheStore(Protocol):
    """Persistence abstraction for compact caches (tiered store implements it)."""

    def put(self, key: str, cache: CompactCache, *, tenant: str = "default") -> str: ...
    def get(self, handle: str) -> Optional[CompactCache]: ...
    def has(self, key: str, *, tenant: str = "default") -> Optional[str]: ...
    def evict(self, handle: str) -> None: ...
    def stats(self) -> dict[str, Any]: ...
