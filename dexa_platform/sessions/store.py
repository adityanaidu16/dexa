"""Session state — the durable context bookkeeping (not the KV; the KV lives in the engine/
LMCache store keyed by prefix). In-memory for the MVP; Redis-swappable, same interface.

A session holds the running message list (its growing context), token estimate, tiering
state, and accumulated savings. The KV for that context is materialized in the backend on
first turn and restored on subsequent turns.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


def _sid() -> str:
    import secrets
    return "sess_" + secrets.token_hex(8)


def est_tokens(messages: list[dict]) -> int:
    # ~4 chars/token over all textual content (good enough for tiering/telemetry)
    n = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    n += len(part.get("text", ""))
    return max(1, n // 4)


@dataclass
class Session:
    id: str
    model: str
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turns: int = 0
    tokens: int = 0
    tier: str = "warm"
    saved_usd: float = 0.0
    saved_ms: float = 0.0

    def public(self) -> dict:
        return {"id": self.id, "model": self.model, "turns": self.turns,
                "tokens": self.tokens, "tier": self.tier,
                "idle_s": round(time.time() - self.last_active, 1),
                "saved_usd": round(self.saved_usd, 6), "saved_ms": round(self.saved_ms, 1),
                "messages": len(self.messages)}


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, Session] = {}

    def create(self, model: str, messages: list[dict] | None = None) -> Session:
        s = Session(id=_sid(), model=model, messages=list(messages or []))
        s.tokens = est_tokens(s.messages)
        with self._lock:
            self._by_id[s.id] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self._by_id.get(sid)

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._by_id.pop(sid, None) is not None

    def list(self) -> list[Session]:
        return list(self._by_id.values())

    def idle_seconds(self, s: Session) -> float:
        return time.time() - s.last_active

    def record_turn(self, s: Session, user_msg: dict, assistant_msg: dict,
                    tier: str, saved: dict) -> None:
        with self._lock:
            s.messages.append(user_msg)
            s.messages.append(assistant_msg)
            s.turns += 1
            s.tokens = est_tokens(s.messages)
            s.tier = tier
            s.saved_usd += saved.get("saved_usd", 0.0)
            s.saved_ms += saved.get("saved_ms", 0.0)
            s.last_active = time.time()
