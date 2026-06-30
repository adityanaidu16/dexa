"""A directory-backed session store: save/load a session's KV state by id.

Minimal on purpose — it models the persistence tier (offload state between turns,
re-attach on any worker). A real deployment backs this with object storage / a
shared NVMe tier; the interface is the same.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from dexa.core.types import KVCache
from dexa.session.state import load_kvcache, save_kvcache


class SessionStore:
    def __init__(self, root: str | Path = ".dexa_sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.npz"

    def save(self, session_id: str, kv: KVCache, *, compress: bool = False) -> dict:
        t0 = time.perf_counter()
        path = save_kvcache(kv, self._path(session_id), compress=compress)
        dt = time.perf_counter() - t0
        return {"session_id": session_id, "path": str(path),
                "save_seconds": dt, "nbytes": int(path.stat().st_size)}

    def load(self, session_id: str) -> tuple[KVCache, float]:
        path = self._path(session_id)
        t0 = time.perf_counter()
        kv = load_kvcache(path)
        return kv, time.perf_counter() - t0

    def has(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def delete(self, session_id: str) -> None:
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.npz"))
