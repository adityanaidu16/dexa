"""Compute a :class:`RecomputePlan` from (previous cached, new) segmentations.

Pure logic — no model, fully unit-testable. Two modes:

* ``mode="exact"`` — reuse the longest common **prefix** of segments (bit-identical
  KV), recompute everything from the first change onward. The result is provably
  identical to a full prefill of the new context. Reuse extent equals prefix
  caching, but on portable/persisted state and over a structured graph (the basis
  for versioning/branching).
* ``mode="selective"`` — additionally flag segments after the edit whose **content
  is unchanged** but whose **position shifted** as ``REUSE_SHIFTED`` candidates.
  These are what CacheBlend-style selective recomputation targets: reuse their KV
  and recompute only the high-deviation tokens. The plan marks them and reports
  ``exact=False``; the engine decides how many tokens to actually recompute.
"""

from __future__ import annotations

from dexa.segment.model import (
    Action,
    RecomputePlan,
    Segment,
    SegmentedContext,
    SegmentPlanItem,
)


def _common_prefix_len(prev: list[Segment], new: list[Segment]) -> int:
    """Number of leading segments identical in content AND order."""
    n = 0
    for a, b in zip(prev, new):
        if a.content_id == b.content_id:
            n += 1
        else:
            break
    return n


def plan_incremental(
    prev: SegmentedContext | None,
    new: SegmentedContext,
    *,
    mode: str = "exact",
) -> RecomputePlan:
    """Plan the minimal work to build ``new``'s KV given ``prev``'s cached KV.

    ``prev=None`` (cold start) => recompute everything. See module docstring for
    the two modes."""
    if mode not in ("exact", "selective"):
        raise ValueError(f"unknown mode {mode!r}; expected 'exact' or 'selective'")

    new_offsets = new.offsets()
    items: list[SegmentPlanItem] = []

    if prev is None:
        for i, seg in enumerate(new.segments):
            start = new_offsets[i]
            items.append(SegmentPlanItem(
                index=i, action=Action.RECOMPUTE, segment=seg,
                new_span=(start, start + seg.n_tokens)))
        return RecomputePlan(items=items, total_tokens=new.n_tokens, exact=True,
                             meta={"mode": mode, "cold_start": True})

    prefix = _common_prefix_len(prev.segments, new.segments)
    prev_offsets = prev.offsets()

    # map content_id -> prev indices lying AFTER the common prefix (shift candidates).
    shifted_pool: dict[str, list[int]] = {}
    for j in range(prefix, len(prev.segments)):
        shifted_pool.setdefault(prev.segments[j].content_id, []).append(j)

    used_exact = False
    exact = True
    for i, seg in enumerate(new.segments):
        start = new_offsets[i]
        span = (start, start + seg.n_tokens)
        if i < prefix:
            items.append(SegmentPlanItem(
                index=i, action=Action.REUSE_EXACT, segment=seg, new_span=span,
                prev_index=i, position_shift=0))
            used_exact = True
            continue

        matched = None
        if mode == "selective":
            pool = shifted_pool.get(seg.content_id)
            if pool:
                matched = pool.pop(0)

        if matched is not None:
            shift = start - prev_offsets[matched]
            items.append(SegmentPlanItem(
                index=i, action=Action.REUSE_SHIFTED, segment=seg, new_span=span,
                prev_index=matched, position_shift=shift))
            exact = False
        else:
            items.append(SegmentPlanItem(
                index=i, action=Action.RECOMPUTE, segment=seg, new_span=span))

    return RecomputePlan(
        items=items, total_tokens=new.n_tokens, exact=exact,
        meta={"mode": mode, "prefix_segments": prefix, "used_exact_prefix": used_exact},
    )
