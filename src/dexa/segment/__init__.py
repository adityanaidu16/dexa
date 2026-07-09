"""Mutable, versioned KV state substrate: segment model + recompute planning."""

from dexa.segment.model import (
    Action,
    RecomputePlan,
    Segment,
    SegmentedContext,
    SegmentPlanItem,
)
from dexa.segment.plan import plan_incremental

__all__ = [
    "Action",
    "RecomputePlan",
    "Segment",
    "SegmentedContext",
    "SegmentPlanItem",
    "plan_incremental",
]
