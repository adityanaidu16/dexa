"""Session API — the stateful product surface.

    POST   /v1/sessions            create a session (optionally with initial context)
    POST   /v1/sessions/{id}/turn  run a turn; KV restored (warm) instead of re-prefilled
    GET    /v1/sessions/{id}       session state + accumulated savings
    GET    /v1/sessions            list
    DELETE /v1/sessions/{id}       drop the session (frees its KV)

Each turn returns telemetry: whether the KV was restored (warm) or prefilled (cold), the
tiering decision (warm/ram/nvme/drop) with its break-even rationale, and the estimated
savings vs a stateless re-prefill of the same context.

    DEXA_SESSION_BACKEND=https://<vllm-lmcache-url>  uvicorn dexa_platform.sessions.service:app --port 8070
    DEXA_SESSION_MOCK=1                               # GPU-free (simulated from measured profile)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import backend, tiering
from .store import SessionStore, est_tokens

app = FastAPI(title="Dexa Sessions", version="0.1.0")
store = SessionStore()
policy = tiering.TieringPolicy()


class CreateIn(BaseModel):
    model: str = "dexa-cua-vlm"
    system: str | None = None
    context: str | None = None          # a big initial context (corpus / conversation seed)


class TurnIn(BaseModel):
    content: str


@app.get("/health")
def health():
    return {"ok": True, "backend": "mock" if backend.MOCK else backend.BACKEND_URL}


@app.post("/v1/sessions")
def create(body: CreateIn):
    msgs: list[dict] = []
    if body.system:
        msgs.append({"role": "system", "content": body.system})
    if body.context:
        msgs.append({"role": "user", "content": body.context})
        msgs.append({"role": "assistant", "content": "Context loaded. Ready."})
    s = store.create(body.model, msgs)
    return {"session": s.public(),
            "note": "first turn prefills the context (cold); subsequent turns restore it (warm)"}


@app.post("/v1/sessions/{sid}/turn")
async def turn(sid: str, body: TurnIn):
    s = store.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")

    idle_s = store.idle_seconds(s)
    user_msg = {"role": "user", "content": body.content}
    full = s.messages + [user_msg]
    tokens = est_tokens(full)
    first_touch = s.turns == 0

    decision = policy.decide(idle_s, tokens)
    try:
        res = await backend.run_turn(full, tokens, first_touch=first_touch)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"backend error: {e}")

    saved = tiering.estimated_savings(tokens, warm=res.warm)
    assistant_msg = {"role": "assistant", "content": res.content}
    store.record_turn(s, user_msg, assistant_msg, tier=decision.tier, saved=saved)

    return {
        "content": res.content,
        "session": s.public(),
        "turn": {
            "warm": res.warm,
            "state": "restored (warm)" if res.warm else "prefilled (cold)",
            "latency_ms": round(res.latency_ms, 1),
            "context_tokens": tokens,
        },
        "tiering": {"tier": decision.tier, "idle_s": round(idle_s, 1),
                    "reason": decision.reason,
                    "breakevens_s": {"warm": round(decision.breakevens.warm_s),
                                     "ram": round(decision.breakevens.ram_s),
                                     "nvme": round(decision.breakevens.nvme_s)}},
        "savings_vs_stateless": saved,
    }


@app.get("/v1/sessions/{sid}")
def get(sid: str):
    s = store.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    return s.public()


@app.get("/v1/sessions")
def list_sessions():
    return {"sessions": [s.public() for s in store.list()]}


@app.delete("/v1/sessions/{sid}")
def delete(sid: str):
    if not store.delete(sid):
        raise HTTPException(404, "session not found")
    return {"deleted": sid}
