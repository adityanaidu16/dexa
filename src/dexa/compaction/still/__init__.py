"""STILL: amortized KV cache compaction in a single forward pass.

A small per-layer Perceiver, trained once against a frozen base model, maps a
full KV cache to a compact one in one forward pass -- the amortized counterpart
to Attention Matching's per-context numerical fit.

torch lives entirely inside this subpackage; it is imported lazily so the rest of
dexa stays torch-free.
"""

from dexa.compaction.still.compactor import StillCompactor
from dexa.compaction.still.perceiver import StillPerceiver

__all__ = ["StillCompactor", "StillPerceiver"]
