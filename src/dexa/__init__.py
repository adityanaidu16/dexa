"""Dexa — compaction-first inference-state engine.

Turns ephemeral KV cache into persistent, compact, versioned, governed state.

Public surface:
    from dexa import KVCache, CompactCache, ModelSpec, CostModel
    from dexa import ModelBackend, FakeBackend
    from dexa import Compactor
"""

from dexa.core.types import (
    ModelSpec,
    LayerKV,
    KVCache,
    CompactLayer,
    CompactCache,
    RefQueries,
    CostModel,
)
from dexa.engine.base import ModelBackend, ContextCache
from dexa.engine.fake import FakeBackend
from dexa.compaction.base import Compactor, CompactionBudget

__all__ = [
    "ModelSpec",
    "LayerKV",
    "KVCache",
    "CompactLayer",
    "CompactCache",
    "RefQueries",
    "CostModel",
    "ModelBackend",
    "ContextCache",
    "FakeBackend",
    "Compactor",
    "CompactionBudget",
]

__version__ = "0.1.0"
