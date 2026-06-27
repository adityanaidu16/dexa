"""Compactor interface.

A compactor maps a full :class:`KVCache` (plus optional reference queries) to a
:class:`CompactCache`. Attention Matching is the flagship; selection methods
(recent-window, heavy-hitter, random) are baselines implemented against the same
interface so the benchmark is apples-to-apples.

The ``budget`` is expressed as a target compression ratio or an absolute number
of compact tokens per kv-head; implementations honor whichever is given.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from dexa.core.types import CompactCache, KVCache, RefQueries


@dataclass
class CompactionBudget:
    """Target size. Exactly one of ``ratio`` / ``tokens_per_head`` is used;
    ``ratio`` takes precedence if both set. ``ratio`` is T/t (e.g. 50 -> 50x)."""

    ratio: Optional[float] = None
    tokens_per_head: Optional[int] = None
    min_tokens: int = 1

    def target_t(self, seq_len: int) -> int:
        if self.ratio is not None:
            return max(self.min_tokens, round(seq_len / self.ratio))
        if self.tokens_per_head is not None:
            return max(self.min_tokens, self.tokens_per_head)
        raise ValueError("CompactionBudget needs ratio or tokens_per_head")


class Compactor(ABC):
    """Produces a compact KV cache from a full one."""

    #: short stable id used in benchmark tables / cache keys
    name: str = "compactor"

    #: whether this method consumes reference queries (attention-matching does;
    #: pure selection baselines may not)
    needs_ref_queries: bool = False

    @abstractmethod
    def compact(
        self,
        cache: KVCache,
        budget: CompactionBudget,
        *,
        ref_queries: Optional[RefQueries] = None,
    ) -> CompactCache: ...
