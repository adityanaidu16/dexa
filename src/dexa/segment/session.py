"""A live, mutable, versioned context — the product surface over the Phase 1
substrate (README "Versioning & branching").

A :class:`SegmentedSession` holds an ordered segmentation and its KV together, and
exposes the operations an agent harness actually needs:

* **mutate** — ``append`` / ``edit`` / ``delete`` a segment. Each mutation updates
  the KV via exact incremental recompute (`HFBackend.recompute_incremental`): the
  unchanged segment prefix is reused, only from the first change onward is
  recomputed. The result is behaviorally identical to a full re-prefill.
* **version** — ``commit(label)`` snapshots the current (segments, KV);
  ``rollback(label)`` restores it. Lets an agent undo a bad turn / speculative edit
  without re-encoding anything.
* **branch** — ``branch()`` forks the session by **copying the existing KV**, not
  by re-prefilling the shared context. Two sub-agents (or speculative paths) then
  diverge independently; the fork costs a memory copy, not the prefill compute.
* **persist** — ``save`` / ``load`` via a :class:`~dexa.session.store.SessionStore`
  (the portable blob format), so a session survives a restart / moves across
  replicas (the persistence wedge).

This is engine-agnostic in shape; today it drives the HF backend (the one with a
real ``recompute_incremental``). It composes the pieces — segment DAG, incremental
recompute, portable persistence — into one object an application can hold.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from dexa.core.types import KVCache
from dexa.segment.model import Segment, SegmentedContext


@dataclass
class _Version:
    segments: tuple[Segment, ...]
    kv: KVCache


def _copy_kv(kv: Optional[KVCache]) -> Optional[KVCache]:
    """Deep-ish copy: numpy arrays copied so branches/snapshots don't alias."""
    if kv is None:
        return None
    from dexa.core.types import LayerKV
    layers = [LayerKV(key=l.key.copy(), value=l.value.copy()) for l in kv.layers]
    return KVCache(spec=kv.spec, layers=layers, positions=kv.positions.copy(),
                   token_ids=list(kv.token_ids) if kv.token_ids is not None else None,
                   meta=dict(kv.meta))


def _torch_capable(backend) -> bool:
    """Whether ``backend`` can hold a GPU-resident cache (an HFBackend-like object
    with a torch model). FakeBackend and friends fall back to the numpy path."""
    if not (hasattr(backend, "model") and hasattr(backend, "device")):
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class SegmentedSession:
    """A live mutable context. When the backend can hold a GPU-resident cache the
    session keeps its KV **on-device** across mutations (only the changed delta is
    recomputed, never round-tripped through host memory — see
    :class:`~dexa.engine.torch_session.TorchKVSession`), and materializes a portable
    numpy :class:`KVCache` only at the persistence boundary (``save``/``commit``).
    Otherwise it uses the numpy incremental path. ``on_device=`` overrides the
    auto-detect."""

    def __init__(self, backend, segments=(), *, on_device: Optional[bool] = None):
        self.be = backend
        self.ctx = SegmentedContext(list(segments))
        self.on_device = _torch_capable(backend) if on_device is None else on_device
        self._dev = None          # TorchKVSession, when on_device
        self._kv: Optional[KVCache] = None   # numpy KVCache (fallback path or materialized)
        self._versions: dict[str, _Version] = {}
        #: cumulative recompute accounting across mutations (the headline savings).
        self.stats = {"mutations": 0, "tokens_recomputed": 0, "tokens_full_reprefill": 0}
        if segments:
            if self.on_device:
                from dexa.engine.torch_session import TorchKVSession
                self._dev = TorchKVSession(backend, self.ctx.token_ids)
            else:
                self._kv = backend.prefill(self.ctx.token_ids)

    # --- introspection ----------------------------------------------------
    @property
    def kv(self) -> Optional[KVCache]:
        """The current KV as a portable numpy :class:`KVCache`. On-device sessions
        materialize lazily and cache until the next mutation."""
        if self.on_device:
            if self._kv is None and self._dev is not None:
                self._kv = self._dev.to_kvcache()
            return self._kv
        return self._kv

    @property
    def segments(self) -> list[Segment]:
        return list(self.ctx.segments)

    @property
    def n_tokens(self) -> int:
        return self.ctx.n_tokens

    def _index_of(self, name_or_index) -> int:
        if isinstance(name_or_index, int):
            return name_or_index
        for i, s in enumerate(self.ctx.segments):
            if s.name == name_or_index:
                return i
        raise KeyError(f"no segment named {name_or_index!r}")

    # --- the mutation core ------------------------------------------------
    def _apply(self, new_ctx: SegmentedContext) -> dict:
        """Update the resident KV to ``new_ctx`` via exact incremental recompute and
        record the tokens-saved accounting."""
        cold = (self._dev is None and self._kv is None) or not self.ctx.segments
        if self.on_device:
            from dexa.engine.torch_session import TorchKVSession
            if cold:
                self._dev = TorchKVSession(self.be, new_ctx.token_ids)
                stats = {"reused_tokens": 0, "recomputed_tokens": new_ctx.n_tokens,
                         "total_tokens": new_ctx.n_tokens}
            else:
                stats = self._dev.apply(self.ctx, new_ctx)
        else:
            if cold:
                self._kv = self.be.prefill(new_ctx.token_ids)
                stats = {"reused_tokens": 0, "recomputed_tokens": new_ctx.n_tokens,
                         "total_tokens": new_ctx.n_tokens}
            else:
                self._kv, stats = self.be.recompute_incremental(self._kv, self.ctx, new_ctx)
        self.ctx = new_ctx
        self._kv = None if self.on_device else self._kv   # invalidate materialized copy
        self.stats["mutations"] += 1
        self.stats["tokens_recomputed"] += stats["recomputed_tokens"]
        self.stats["tokens_full_reprefill"] += new_ctx.n_tokens
        return stats

    def append(self, segment: Segment) -> dict:
        return self._apply(SegmentedContext(self.ctx.segments + [segment]))

    def edit(self, name_or_index, new_segment: Segment) -> dict:
        i = self._index_of(name_or_index)
        segs = list(self.ctx.segments)
        segs[i] = new_segment
        return self._apply(SegmentedContext(segs))

    def delete(self, name_or_index) -> dict:
        i = self._index_of(name_or_index)
        segs = list(self.ctx.segments)
        del segs[i]
        return self._apply(SegmentedContext(segs))

    # --- versioning -------------------------------------------------------
    def commit(self, label: str) -> None:
        """Snapshot the current (segments, KV) under ``label`` (copy-on-write)."""
        self._versions[label] = _Version(tuple(self.ctx.segments), _copy_kv(self.kv))

    def rollback(self, label: str) -> None:
        """Restore a previously committed snapshot — no recompute."""
        if label not in self._versions:
            raise KeyError(f"no committed version {label!r}")
        v = self._versions[label]
        self.ctx = SegmentedContext(list(v.segments))
        if self.on_device:
            from dexa.engine.torch_session import TorchKVSession
            self._dev = TorchKVSession.from_kvcache(self.be, v.kv)
            self._kv = None
        else:
            self._kv = _copy_kv(v.kv)

    def versions(self) -> list[str]:
        return sorted(self._versions)

    def branch(self) -> "SegmentedSession":
        """Fork: a new session with a **copy** of this KV (no re-prefill of the
        shared context — on-device this clones the resident tensors). The two
        diverge independently."""
        child = SegmentedSession(self.be, segments=(), on_device=self.on_device)
        child.ctx = SegmentedContext(list(self.ctx.segments))
        if self.on_device:
            child._dev = self._dev.clone() if self._dev is not None else None
        else:
            child._kv = _copy_kv(self._kv)
        return child

    def diff(self, other: "SegmentedSession") -> dict:
        """Segment-level diff vs another session: matching prefix + per-side tails."""
        a, b = self.ctx.segments, other.ctx.segments
        prefix = 0
        for x, y in zip(a, b):
            if x.content_id == y.content_id:
                prefix += 1
            else:
                break
        return {
            "common_prefix_segments": prefix,
            "only_self": [s.name for s in a[prefix:]],
            "only_other": [s.name for s in b[prefix:]],
        }

    # --- persistence ------------------------------------------------------
    def save(self, store, session_id: str, **kw) -> dict:
        if self.kv is None:
            raise ValueError("nothing to persist (empty session)")
        return store.save(session_id, self.kv, **kw)

    @classmethod
    def load(cls, backend, store, session_id: str, segments, *,
             on_device: Optional[bool] = None) -> "SegmentedSession":
        """Reattach a persisted session: load its KV and pair it with the known
        segmentation (positions/tokens must match what was saved)."""
        s = cls(backend, segments=(), on_device=on_device)
        s.ctx = SegmentedContext(list(segments))
        kv, _ = store.load(session_id)
        if kv.seq_len != s.ctx.n_tokens:
            raise ValueError("loaded KV length does not match the provided segmentation")
        if s.on_device:
            from dexa.engine.torch_session import TorchKVSession
            s._dev = TorchKVSession.from_kvcache(backend, kv)
        else:
            s._kv = kv
        return s
