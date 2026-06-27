"""Bounded, iteratively-compacted working memory for long-horizon agents.

The thesis
----------
A long-running agent keeps appending context (turns, tool outputs, retrieved
docs). The raw KV cache grows without bound -- the *memory wall*. Naive fixes
(hard recent-window truncation) keep memory bounded but forget early facts.
:class:`WorkingMemory` instead keeps a **bounded** working set:

    combined cache  =  [ compact_memory ; recent_raw ]

where ``recent_raw`` is the most recent ``keep_recent_tokens`` of context kept
*verbatim* and ``compact_memory`` is everything older, compressed (by any
:class:`~dexa.compaction.base.Compactor`) to a fixed budget. As the trajectory
grows, the *same* fixed compact budget is re-used to hold an ever-larger logical
span -- that is the iterative working-memory compaction.

Why we rebuild the compact memory from a prefix prefill
-------------------------------------------------------
A KV cache is **not** composable by independently prefilling chunks: a chunk
prefilled in isolation never attended to the context before it, so its keys/values
are wrong. Causality, however, gives us a clean handle: with causal attention,
token *i*'s KV depends only on tokens ``[0, i]``, so prefilling a **prefix**
``[0, m)`` yields *exactly* the same KV for those *m* tokens as prefilling the
whole sequence. We exploit this:

* ``compact_memory`` is produced by compacting ``prefill(tokens[:compact_end])``
  -- a correct prefix KV -- with reference queries from the *same* tokens (the
  exact, proven recipe used in ``benchmarks/niah_real.py``). So early facts are
  faithfully represented.
* ``recent_raw`` is one block prefill of ``tokens[compact_end:total]`` at its
  true absolute position offset, so the recent window's keys attend to each other
  and carry correct absolute RoPE. (It does not re-attend to the already-compacted
  prefix; that small approximation is the price of not retaining old raw KV.)

Positions / RoPE
----------------
Keys keep their absolute RoPE phase end-to-end: the compact prefix lives at
``[0, compact_end)`` and the recent block at ``[compact_end, total)``. The fused
cache decodes a new query at position ``total`` (the backend reads
:attr:`CompactCache.logical_length`), so RoPE is consistent across the boundary.

Memory vs. compute
------------------
The *maintained* working set is bounded: ``current_tokens = compact_phys +
recent_tokens <= budget_tokens`` after every append. Compaction pays a
**recompute** cost -- it re-prefills the older prefix to recompress it (the prefix
KV is transient and immediately discarded, never persisted). This is the usual
memory-vs-compute trade of compression-based memory (read the history to compress
it, then keep only the small summary). We surface the transient recompute size in
:meth:`stats` (``peak_recompute_tokens``) and report it honestly in the benchmark.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dexa.compaction.base import CompactionBudget, Compactor
from dexa.core.types import CompactCache, CompactLayer, KVCache
from dexa.engine.base import ContextCache, ModelBackend


# --- immutable versioning handle ------------------------------------------
@dataclass(frozen=True)
class MemorySnapshot:
    """An immutable handle to a :class:`WorkingMemory`'s compact state.

    A lightweight versioning primitive: it deep-copies the :class:`CompactCache`
    so later mutation of the live memory cannot change the snapshot. Enough to
    show the "fork / checkpoint a session's working memory" capability.
    """

    compact: Optional[CompactCache]
    compact_end: int      # logical tokens absorbed into the compact memory
    total: int            # total logical tokens appended so far
    recent_tokens: int    # raw tokens still held verbatim at snapshot time
    version: int          # bumps on every compaction / commit

    @property
    def compact_budget(self) -> int:
        return 0 if self.compact is None else self.compact.budget


class WorkingMemory:
    """Maintain a bounded compact KV state across an agent trajectory.

    Parameters
    ----------
    backend:
        The model backend (prefill, reference queries, decode/score).
    compactor:
        Any :class:`~dexa.compaction.base.Compactor` (attention matching, heavy
        hitter, recent window, ...). It compresses the older context.
    budget_tokens:
        The maximum maintained working set, in per-head KV tokens.
    keep_recent_tokens:
        How many of the most recent tokens to keep raw (verbatim). Must be
        strictly less than ``budget_tokens``; the remainder
        ``budget_tokens - keep_recent_tokens`` is the compact memory's budget.
    ref_strategy:
        Reference-query strategy forwarded to ``backend.reference_queries``.
    ref_per_head:
        Per-head cap on reference queries used during compaction (cost control).
    """

    def __init__(
        self,
        backend: ModelBackend,
        compactor: Compactor,
        budget_tokens: int,
        keep_recent_tokens: int,
        ref_strategy: str = "repeat_prefill",
        ref_per_head: int = 128,
    ) -> None:
        if keep_recent_tokens >= budget_tokens:
            raise ValueError("keep_recent_tokens must be < budget_tokens")
        if keep_recent_tokens < 0:
            raise ValueError("keep_recent_tokens must be >= 0")

        self.backend = backend
        self.compactor = compactor
        self.budget_tokens = int(budget_tokens)
        self.keep_recent_tokens = int(keep_recent_tokens)
        self.compact_budget = self.budget_tokens - self.keep_recent_tokens
        self.ref_strategy = ref_strategy
        self.ref_per_head = int(ref_per_head)
        self.spec = backend.spec

        # live state
        self.all_tokens: list[int] = []
        self.compact_mem: Optional[CompactCache] = None
        self.recent_raw: Optional[KVCache] = None
        self.compact_end: int = 0   # logical tokens folded into compact_mem
        self.total: int = 0         # total logical tokens appended

        # stats
        self.n_compactions: int = 0
        self.peak_tokens: int = 0
        self.peak_recompute_tokens: int = 0  # transient prefix re-prefill size
        self.compute_seconds: float = 0.0
        self._version: int = 0

    # --- size accounting --------------------------------------------------
    @property
    def n_recent(self) -> int:
        return 0 if self.recent_raw is None else self.recent_raw.seq_len

    @property
    def compact_phys(self) -> int:
        """Per-head physical tokens held by the compact memory (uniform across
        heads/layers for the supported compactors)."""
        if self.compact_mem is None:
            return 0
        return int(self.compact_mem.layers[0].keys[0].shape[0])

    @property
    def current_tokens(self) -> int:
        """Maintained working set in per-head KV tokens (compact + raw recent)."""
        return self.compact_phys + self.n_recent

    # --- the trajectory API ----------------------------------------------
    def append(self, token_ids: list[int]) -> None:
        """Append a new chunk, refresh the raw recent window, and compact the
        oldest tokens if the working set would exceed the budget."""
        token_ids = list(token_ids)
        if not token_ids:
            return
        t0 = time.perf_counter()
        self.all_tokens.extend(token_ids)
        self.total = len(self.all_tokens)
        self._refresh_recent()
        self.compute_seconds += time.perf_counter() - t0

        self.peak_tokens = max(self.peak_tokens, self.current_tokens)
        self._maybe_compact()
        self.peak_tokens = max(self.peak_tokens, self.current_tokens)

    def query(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int = 32,
        greedy: bool = True,
        score_targets: Optional[list[int]] = None,
    ) -> "np.ndarray | list[int]":
        """Decode (or score) against the current combined cache.

        With ``score_targets`` set, returns teacher-forced per-token log-probs of
        the targets (delegates to ``backend.score``); otherwise greedily
        generates a continuation (delegates to ``backend.generate``).
        """
        ctx = self.combined()
        if score_targets is not None:
            return self.backend.score(ctx, list(prompt_ids), list(score_targets))
        return self.backend.generate(
            ctx, list(prompt_ids), max_new_tokens=max_new_tokens, greedy=greedy
        )

    # --- versioning -------------------------------------------------------
    def snapshot(self) -> MemorySnapshot:
        """Return an immutable, independent handle to the current compact state."""
        return MemorySnapshot(
            compact=copy.deepcopy(self.compact_mem),
            compact_end=self.compact_end,
            total=self.total,
            recent_tokens=self.n_recent,
            version=self._version,
        )

    def commit(self) -> MemorySnapshot:
        """Alias for :meth:`snapshot` that also bumps the version counter, so a
        sequence of commits yields monotonically-increasing handles."""
        self._version += 1
        return self.snapshot()

    # --- stats ------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "current_tokens": self.current_tokens,
            "peak_tokens": self.peak_tokens,
            "n_compactions": self.n_compactions,
            "compact_tokens": self.compact_phys,
            "recent_tokens": self.n_recent,
            "total_logical_tokens": self.total,
            "compute_seconds": self.compute_seconds,
            "peak_recompute_tokens": self.peak_recompute_tokens,
            "version": self._version,
        }

    # --- internals --------------------------------------------------------
    def _refresh_recent(self) -> None:
        """(Re)prefill the recent window ``[compact_end, total)`` as one block at
        its true absolute position offset."""
        recent_tokens = self.all_tokens[self.compact_end : self.total]
        if not recent_tokens:
            self.recent_raw = None
            return
        self.recent_raw = self.backend.prefill(
            recent_tokens, position_offset=self.compact_end
        )

    def _maybe_compact(self) -> None:
        # One pass normally restores current_tokens <= budget; the loop guards a
        # single oversized append.
        while self.current_tokens > self.budget_tokens and (
            self.total - self.compact_end
        ) > self.keep_recent_tokens:
            before = self.compact_end
            self._compact_step()
            if self.compact_end <= before:  # safety: no progress
                break

    def _compact_step(self) -> None:
        """Recompress everything older than the recent window into the fixed
        compact budget, rebuilt from a correct prefix prefill."""
        t0 = time.perf_counter()
        new_end = self.total - self.keep_recent_tokens
        old_tokens = self.all_tokens[:new_end]
        self.peak_recompute_tokens = max(self.peak_recompute_tokens, len(old_tokens))

        full_old = self.backend.prefill(old_tokens)  # prefix KV: exact by causality
        refs = None
        if self.compactor.needs_ref_queries:
            refs = self.backend.reference_queries(
                old_tokens, strategy=self.ref_strategy, n_per_head=self.ref_per_head
            )
        budget = CompactionBudget(tokens_per_head=self.compact_budget)
        compact = self.compactor.compact(full_old, budget, ref_queries=refs)
        # logical_length already equals len(old_tokens) == new_end (absolute).
        compact.meta = dict(compact.meta or {})
        # Sanitize compact values to the legitimate raw-value scale. Some value
        # fits (notably attention matching's least-squares step) leave keys with
        # near-zero attention weight at astronomically large values; harmless in a
        # self-contained softmax (~0 weight) but they poison the fused
        # [compact ; recent] attention output, where any non-zero weight times a
        # huge value dominates. Clipping bounds that contribution without changing
        # well-attended keys.
        self._clip_values(compact, full_old)

        self.compact_mem = compact
        self.compact_end = new_end
        self._refresh_recent()
        self.n_compactions += 1
        self._version += 1
        self.compute_seconds += time.perf_counter() - t0

    @staticmethod
    def _clip_values(compact: CompactCache, full: KVCache) -> None:
        """Clip compact values to the max-abs raw value magnitude (per layer)."""
        for cl, fl in zip(compact.layers, full.layers):
            vmax = float(np.max(np.abs(fl.value))) if fl.value.size else 0.0
            if vmax <= 0.0:
                continue
            for h in range(len(cl.values)):
                np.clip(cl.values[h], -vmax, vmax, out=cl.values[h])

    def combined(self) -> ContextCache:
        """Build the single cache used for decode: ``[compact_mem ; recent_raw]``.

        Returns the raw cache directly when nothing has been compacted yet (so
        the no-compaction path is exact -- it is a correct full prefix prefill),
        the compact cache when nothing recent remains, and otherwise a fused
        :class:`CompactCache` with ``logical_length = total``.
        """
        if self.compact_mem is None:
            if self.recent_raw is None:
                raise ValueError("empty working memory: nothing to query")
            return self.recent_raw  # compact_end == 0 -> absolute positions from 0
        if self.recent_raw is None or self.recent_raw.seq_len == 0:
            return self.compact_mem

        s = self.spec
        cm, rr = self.compact_mem, self.recent_raw
        nrec = rr.seq_len
        layers: list[CompactLayer] = []
        for l in range(s.n_layers):
            cl, rl = cm.layers[l], rr.layers[l]
            keys, values, biases, poss = [], [], [], []
            for h in range(s.n_kv_heads):
                keys.append(np.concatenate([cl.keys[h], rl.key[h]], axis=0).astype(np.float32))
                values.append(np.concatenate([cl.values[h], rl.value[h]], axis=0).astype(np.float32))
                biases.append(
                    np.concatenate([cl.biases[h], np.zeros(nrec, dtype=np.float32)]).astype(
                        np.float32
                    )
                )
                poss.append(np.concatenate([cl.positions[h], rr.positions]).astype(np.int64))
            layers.append(CompactLayer(keys=keys, values=values, biases=biases, positions=poss))
        return CompactCache(
            spec=s,
            layers=layers,
            logical_length=self.total,
            method="working_memory",
            meta={"compact_end": self.compact_end},
        )
