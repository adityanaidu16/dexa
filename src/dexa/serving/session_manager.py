"""Engine-agnostic stateful-session lifecycle.

A session is a growing KV cache. Each turn prefills only the *delta* (the new
user turn) against the cached context — not the whole history — then decodes,
folds the response into the cache, and persists. The session can be restored on
any worker after a restart (load from the store). This is the logic the vLLM
connector mirrors over paged KV; here it runs on any ModelBackend (CPU-validated
via the HF backend, and usable as a standalone library).

Token format: this reference core uses a simple, consistent ``System:/User:/
Assistant:`` framing tracked at the token level (append-only, so there is no
re-tokenization drift). A production integration wires the model's native chat
template with token-level diffing — same lifecycle, better instruct formatting.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from dexa.core.types import KVCache
from dexa.session.store import SessionStore


@dataclass
class SessionInfo:
    session_id: str
    turns: int
    context_tokens: int        # total tokens now in the session KV
    prefill_delta_tokens: int  # tokens prefilled THIS turn (the delta)
    stateless_would_prefill: int  # tokens a stateless engine re-prefills each turn
    resumed: bool              # was the session loaded from the store this call?
    state_bytes: int

    @property
    def prefill_savings(self) -> float:
        """Fraction of prefill avoided vs a stateless engine this turn."""
        if self.stateless_would_prefill <= 0:
            return 0.0
        return 1.0 - self.prefill_delta_tokens / self.stateless_would_prefill


class SessionManager:
    def __init__(self, backend, *, store: Optional[SessionStore] = None,
                 persist: bool = True, keep_resident: int = 16) -> None:
        self.backend = backend
        self.store = store or SessionStore()
        self.persist = persist
        self.keep_resident = keep_resident
        self._resident: "OrderedDict[str, KVCache]" = OrderedDict()

    # --- cache residency / persistence -----------------------------------
    def _get(self, session_id: str) -> tuple[Optional[KVCache], bool]:
        if session_id in self._resident:
            self._resident.move_to_end(session_id)
            return self._resident[session_id], False
        if self.persist and self.store.has(session_id):
            kv, _ = self.store.load(session_id)          # restore after restart
            self._resident[session_id] = kv
            return kv, True
        return None, False

    def _put(self, session_id: str, kv: KVCache) -> int:
        self._resident[session_id] = kv
        self._resident.move_to_end(session_id)
        nbytes = 0
        if self.persist:
            nbytes = self.store.save(session_id, kv)["nbytes"]
        while len(self._resident) > self.keep_resident:
            self._resident.popitem(last=False)           # evict LRU (already persisted)
        return nbytes or kv.nbytes()

    # --- token framing (append-only) -------------------------------------
    def _turn_tokens(self, user_text: str, *, system: Optional[str], first: bool) -> list[int]:
        prefix = f"System: {system}\n\n" if (first and system) else ""
        return self.backend.tokenize(f"{prefix}User: {user_text}\n\nAssistant:")

    @staticmethod
    def _clean(text: str) -> str:
        # stop the response at a hallucinated next turn, if any.
        for marker in ("\nUser:", "\nSystem:", "User:"):
            i = text.find(marker)
            if i != -1:
                text = text[:i]
        return text.strip()

    # --- the API ----------------------------------------------------------
    def turn(self, session_id: str, user_text: str, *, system: Optional[str] = None,
             max_new_tokens: int = 128) -> tuple[str, SessionInfo]:
        """Run one turn: prefill the delta, decode, fold in, persist. Returns
        (assistant_text, info)."""
        kv, resumed = self._get(session_id)
        turns = (kv.meta.get("turns", 0) if kv is not None else 0) + 1

        if kv is None:
            delta = self._turn_tokens(user_text, system=system, first=True)
            kv0 = self.backend.prefill(delta[:1])
            delta_rest = delta[1:]
            stateless_would = len(delta)
        else:
            delta_rest = self._turn_tokens(user_text, system=None, first=False)
            kv0 = kv
            stateless_would = kv.seq_len + len(delta_rest)

        resp_tokens, new_kv = self.backend.generate_and_extend(
            kv0, delta_rest, max_new_tokens=max_new_tokens)
        new_kv.meta["turns"] = turns
        state_bytes = self._put(session_id, new_kv)

        text = self._clean(self.backend.detokenize(resp_tokens))
        info = SessionInfo(
            session_id=session_id, turns=turns, context_tokens=new_kv.seq_len,
            prefill_delta_tokens=len(delta_rest) + (1 if kv is None else 0),
            stateless_would_prefill=stateless_would, resumed=resumed,
            state_bytes=state_bytes,
        )
        return text, info

    def exists(self, session_id: str) -> bool:
        return session_id in self._resident or (self.persist and self.store.has(session_id))

    def drop(self, session_id: str) -> None:
        self._resident.pop(session_id, None)
        if self.persist:
            self.store.delete(session_id)
