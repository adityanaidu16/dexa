"""A directory-backed session store: save/load a session's KV state by id.

Minimal on purpose — it models the persistence tier (offload state between turns,
re-attach on any worker). A real deployment backs this with object storage / a
shared NVMe tier; the interface is the same.

Two on-disk formats are supported, chosen by ``format=`` on construction:

* ``"npz"`` (default, back-compatible): the numpy ``.npz`` container
  (:mod:`dexa.session.state`).
* ``"blob"``: the memory-mapped binary format (:mod:`dexa.session.blob`) — a
  zero-copy load path that removes the ZIP-parse + full-array copy npz forces on
  every resume (resume latency is the headline metric). Prefer it for the serving
  path; ``persist_demo``/benchmarks can toggle it.

Both formats honor ``precision`` (persist at the model's native dtype — ~2× smaller
state for bf16/fp16 models). Loads **auto-detect** the format by file suffix, so a
store can read either regardless of its configured save format.
"""

from __future__ import annotations

import time
from pathlib import Path

from dexa.core.types import KVCache
from dexa.session.blob import load_kvcache_blob, save_kvcache_blob
from dexa.session.state import load_kvcache, save_kvcache

_SUFFIX = {"npz": ".npz", "blob": ".dexakv"}


class SessionStore:
    def __init__(self, root: str | Path = ".dexa_sessions", *, format: str = "npz") -> None:
        if format not in _SUFFIX:
            raise ValueError(f"unknown format {format!r}; expected 'npz' or 'blob'")
        self.root = Path(root)
        self.format = format
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}{_SUFFIX[self.format]}"

    def _resolve(self, session_id: str) -> Path | None:
        """Find an existing file for ``session_id`` in either format (configured
        format first)."""
        primary = self._path(session_id)
        if primary.exists():
            return primary
        for suf in _SUFFIX.values():
            cand = self.root / f"{session_id}{suf}"
            if cand.exists():
                return cand
        return None

    def save(
        self, session_id: str, kv: KVCache, *, compress: bool = False, precision: str = "auto"
    ) -> dict:
        t0 = time.perf_counter()
        if self.format == "blob":
            path = save_kvcache_blob(kv, self._path(session_id), precision=precision)
        else:
            path = save_kvcache(kv, self._path(session_id), compress=compress, precision=precision)
        dt = time.perf_counter() - t0
        return {"session_id": session_id, "path": str(path), "format": self.format,
                "save_seconds": dt, "nbytes": int(path.stat().st_size)}

    def load(self, session_id: str, *, keep_native: bool = False) -> tuple[KVCache, float]:
        """Load a session's KVCache and the wall-time it took.

        ``keep_native=True`` asks the blob format to skip the bf16→fp32 host widen
        and return layers in their store dtype (see
        :func:`dexa.session.blob.load_kvcache_blob`) — the fast resume path, valid
        only for a dtype-aware consumer like ``HFBackend``. Ignored for the ``.npz``
        format (which always materializes fp32)."""
        path = self._resolve(session_id)
        if path is None:
            raise FileNotFoundError(f"no persisted session {session_id!r} in {self.root}")
        t0 = time.perf_counter()
        if path.suffix == ".dexakv":
            kv = load_kvcache_blob(path, keep_native=keep_native)
        else:
            kv = load_kvcache(path)
        return kv, time.perf_counter() - t0

    def has(self, session_id: str) -> bool:
        return self._resolve(session_id) is not None

    def delete(self, session_id: str) -> None:
        for suf in _SUFFIX.values():
            p = self.root / f"{session_id}{suf}"
            if p.exists():
                p.unlink()

    def list_ids(self) -> list[str]:
        ids = set()
        for suf in _SUFFIX.values():
            ids.update(p.name[: -len(suf)] for p in self.root.glob(f"*{suf}"))
        return sorted(ids)
