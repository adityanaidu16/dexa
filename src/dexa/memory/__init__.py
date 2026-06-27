"""Working-memory: a bounded, iteratively-compacted KV state for long-horizon agents.

A long-running agent accumulates context (tool outputs, retrieved docs, file
reads, prior turns) until the raw KV cache becomes huge -- the *memory wall*.
:class:`~dexa.memory.working_memory.WorkingMemory` maintains a **bounded** working
set by iteratively compacting the oldest context while keeping the most recent
context raw (STILL-style: compress old, keep recent), so memory stays flat and
late-turn recall stays high.
"""

from dexa.memory.working_memory import MemorySnapshot, WorkingMemory

__all__ = ["WorkingMemory", "MemorySnapshot"]
