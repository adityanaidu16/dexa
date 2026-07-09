"""Segment model + causal-dependency recompute planning — the substrate for
mutable, versioned KV state (README Phase 1).

The idea. A model's context is not an opaque token blob; it is an ordered list of
**segments** — a system prompt, tool definitions, retrieved documents, prior
turns, tool results. Each segment has a stable content identity (a hash of its
tokens). When an agent mutates the context — edits a file it pasted, replaces a
stale tool result, drops a document — most segments are untouched. The question
this module answers is precisely: *given the previous (cached) segmentation and
the new one, what is the minimum that must be recomputed?*

Causal-dependency rule. Transformer attention is causal: a token attends to every
token at or before its position. So a segment's KV depends on **all segments
before it**. That gives an exact, defensible dependency graph:

* Segments forming the **longest common prefix** (same content, same order, same
  absolute positions) are reusable **exactly** — bit-identical KV, no work.
* From the first changed segment onward, content and/or absolute positions differ,
  so an *exact* result requires recompute (this is the exactness floor, and it is
  what prefix caching already achieves in reuse extent).
* A segment after the edit whose **content is identical** to a previous segment
  but whose **position shifted** (because an upstream segment changed length) is a
  *reuse candidate*: its tokens' own content is unchanged, only their position and
  their attention over the edited region changed. Reusing it exactly is wrong, but
  it is the target of **selective recomputation** (CacheBlend, EuroSys'25): recompute
  only the small fraction of high-deviation tokens to restore cross-attention. This
  module *identifies* those candidates; the selective mechanism lives in the engine.

This file is **pure** (numpy/stdlib, no torch): the plan is computed from segment
metadata alone, so it is fully unit-testable without a model. The engine consumes
a :class:`RecomputePlan` to actually reuse/recompute KV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dexa.core.types import hash_tokens


@dataclass(frozen=True)
class Segment:
    """One content-identified span of a context.

    ``content_id`` is a stable hash of the tokens (order- and length-sensitive), so
    two segments with the same text on any instance / at any time share an id — the
    key under which a segment's KV is stored and matched.
    """

    name: str
    token_ids: tuple[int, ...]
    role: str = "context"  # advisory: system|tools|doc|turn|tool_result|query|...

    @property
    def content_id(self) -> str:
        return hash_tokens(list(self.token_ids))

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class SegmentedContext:
    """An ordered list of segments = a full context with structure preserved."""

    segments: list[Segment]

    @property
    def token_ids(self) -> list[int]:
        out: list[int] = []
        for s in self.segments:
            out.extend(s.token_ids)
        return out

    @property
    def n_tokens(self) -> int:
        return sum(s.n_tokens for s in self.segments)

    def offsets(self) -> list[int]:
        """Absolute start position of each segment (position ids are contiguous)."""
        offs, acc = [], 0
        for s in self.segments:
            offs.append(acc)
            acc += s.n_tokens
        return offs

    def span(self, i: int) -> tuple[int, int]:
        """[start, end) absolute token range of segment ``i``."""
        offs = self.offsets()
        return offs[i], offs[i] + self.segments[i].n_tokens


class Action(str, Enum):
    REUSE_EXACT = "reuse_exact"        # prefix: identical content AND position
    REUSE_SHIFTED = "reuse_shifted"    # identical content, shifted position -> selective-recompute candidate
    RECOMPUTE = "recompute"            # new/changed content -> must prefill


@dataclass
class SegmentPlanItem:
    """Per-segment decision in the new context."""

    index: int
    action: Action
    segment: Segment
    new_span: tuple[int, int]                 # [start, end) in the NEW context
    prev_index: Optional[int] = None          # matched segment in the prev context (if any)
    position_shift: int = 0                   # new_start - prev_start (0 for exact prefix)


@dataclass
class RecomputePlan:
    """The output: what to reuse vs recompute to build the new context's KV.

    ``exact`` = the plan is bit-identical to a full prefill using only REUSE_EXACT
    + RECOMPUTE (no shifted reuse). ``recompute_ranges`` are the NEW-context token
    ranges the engine must actually run the model over."""

    items: list[SegmentPlanItem]
    total_tokens: int
    exact: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def reused_exact_tokens(self) -> int:
        return sum(it.segment.n_tokens for it in self.items if it.action == Action.REUSE_EXACT)

    @property
    def reuse_shifted_tokens(self) -> int:
        return sum(it.segment.n_tokens for it in self.items if it.action == Action.REUSE_SHIFTED)

    @property
    def recompute_tokens(self) -> int:
        return sum(it.segment.n_tokens for it in self.items if it.action == Action.RECOMPUTE)

    def recompute_ranges(self) -> list[tuple[int, int]]:
        """Merged contiguous NEW-context token ranges to run the model over from
        scratch. Only hard ``RECOMPUTE`` segments — ``REUSE_SHIFTED`` candidates are
        handled by the selective (partial) mechanism, not a full forward."""
        raw = sorted(it.new_span for it in self.items if it.action == Action.RECOMPUTE)
        merged: list[tuple[int, int]] = []
        for a, b in raw:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def savings(self) -> dict:
        """Tokens reprocessed vs a full re-prefill of the new context."""
        full = self.total_tokens
        recomputed = full - self.reused_exact_tokens  # exact plan reprocesses everything past the edit
        return {
            "total_tokens": full,
            "reused_exact_tokens": self.reused_exact_tokens,
            "reuse_shifted_candidate_tokens": self.reuse_shifted_tokens,
            "recomputed_tokens_exact": recomputed,
            "recompute_fraction_exact": (recomputed / full) if full else 0.0,
            "prefix_reuse_fraction": (self.reused_exact_tokens / full) if full else 0.0,
        }
