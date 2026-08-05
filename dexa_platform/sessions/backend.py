"""Session backend — runs a turn against the stateful engine (vLLM + LMCache).

A turn sends the session's FULL running context. On the real backend, LMCache restores the KV
for the already-seen prefix (from GPU/CPU/NVMe) and prefills only the new delta — so the turn
is fast and cheap. The mock simulates this from the measured profile so the whole service is
testable with no GPU.
"""

from __future__ import annotations

import os
import time

import httpx

from . import tiering

BACKEND_URL = os.environ.get("DEXA_SESSION_BACKEND", "").rstrip("/")
BACKEND_MODEL = os.environ.get("DEXA_SESSION_MODEL", "dexa-cua-vlm")
MOCK = os.environ.get("DEXA_SESSION_MOCK") == "1" or not BACKEND_URL


class TurnResult:
    def __init__(self, content: str, latency_ms: float, prompt_tokens: int,
                 completion_tokens: int, warm: bool):
        self.content = content
        self.latency_ms = latency_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.warm = warm            # True if KV was restored (not a first-touch cold prefill)


def _mock_turn(messages: list[dict], tokens: int, warm: bool) -> TurnResult:
    # cold = pay the measured prefill; warm = pay the measured CPU restore. Small jitter-free.
    _kv, prefill_ms, restore_ms, _rn = tiering.profile_for(tokens)
    latency = restore_ms if warm else prefill_ms
    return TurnResult("ACK: " + (messages[-1].get("content", "") if messages else "")[:40],
                      latency_ms=latency, prompt_tokens=tokens, completion_tokens=8, warm=warm)


async def run_turn(messages: list[dict], tokens: int, first_touch: bool) -> TurnResult:
    """first_touch=True means this exact context prefix has never been served -> cold prefill."""
    if MOCK:
        return _mock_turn(messages, tokens, warm=not first_touch)

    payload = {"model": BACKEND_MODEL, "messages": messages, "max_tokens": 16,
               "temperature": 0.0}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{BACKEND_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    latency = (time.perf_counter() - t0) * 1000
    usage = data.get("usage") or {}
    content = data["choices"][0]["message"].get("content", "")
    # warm is deterministic: if we've served this session's prefix before, LMCache restores it.
    # (Latency can't tell them apart end-to-end — network + decode overhead sit in both turns.)
    return TurnResult(content, latency_ms=latency,
                      prompt_tokens=usage.get("prompt_tokens", tokens),
                      completion_tokens=usage.get("completion_tokens", 0), warm=not first_touch)
