"""Dexa gateway — an OpenAI-compatible endpoint for computer-use agents.

Switching cost is two lines: point your OpenAI client's base_url at this server and use your
Dexa key. Every `/v1/chat/completions` call is served by the optimized open VLM backend, and
the response carries — in the body under `dexa` and in `x-dexa-*` headers — exactly what the
call cost here vs what it would have cost on the frontier model you came from, plus the
measured screen redundancy for the session. Immediate, per-request, auditable impact.

    uvicorn dexa_platform.gateway.app:app --port 8080

Env:
    DEXA_BACKEND_URL   OpenAI-compatible VLM backend (e.g. the Modal Qwen2.5-VL endpoint).
                       Unset (or DEXA_MOCK=1) -> mock backend, runs with no GPU.
    DEXA_BACKEND_MODEL served-model-name at the backend (default "doc-vlm").
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import pricing
from .redundancy import RedundancyMeter
from .store import Store

BACKEND_URL = os.environ.get("DEXA_BACKEND_URL", "").rstrip("/")
BACKEND_MODEL = os.environ.get("DEXA_BACKEND_MODEL", "doc-vlm")
MOCK = os.environ.get("DEXA_MOCK") == "1" or not BACKEND_URL
DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")

app = FastAPI(title="Dexa Gateway", version="0.1.0")
meter = RedundancyMeter()
store = Store()
# per-session last (image_hash, text_hash) -> cached response, for exact-duplicate reuse
_cache: dict[str, tuple[str, str, dict]] = {}


def _tok_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # ~4 chars/token; good enough for cost display


_DATA_URI = re.compile(r"^data:image/[^;]+;base64,(.*)$", re.DOTALL)


def _extract(messages: list) -> tuple[list[tuple[int, int]], list[bytes], str]:
    """Return (image dims, image bytes for the *last* image, concatenated text)."""
    dims: list[tuple[int, int]] = []
    all_bytes: list[bytes] = []
    text_parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            text_parts.append(c)
            continue
        for part in c or []:
            t = part.get("type")
            if t == "text":
                text_parts.append(part.get("text", ""))
            elif t == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                mo = _DATA_URI.match(url)
                if mo:
                    try:
                        raw = base64.b64decode(mo.group(1))
                        all_bytes.append(raw)
                        dims.append(_img_dims(raw))
                    except Exception:
                        dims.append((1280, 800))
                else:
                    dims.append((1280, 800))  # remote URL: assume a typical screen
    return dims, all_bytes, "\n".join(text_parts)


def _img_dims(raw: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            return im.size  # (w, h)
    except Exception:
        return (1280, 800)


def _mock_completion(model: str, prompt: str) -> dict:
    reply = "CLICK(640, 384)"  # a plausible computer-use action
    return {
        "id": "chatcmpl-dexa-mock", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": reply}}],
        "usage": {"prompt_tokens": _tok_estimate(prompt), "completion_tokens": _tok_estimate(reply),
                  "total_tokens": _tok_estimate(prompt) + _tok_estimate(reply)},
    }


async def _forward(body: dict) -> dict:
    payload = dict(body)
    payload["model"] = BACKEND_MODEL
    payload["stream"] = False
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{BACKEND_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


@app.get("/healthz")
async def healthz():
    return {"ok": True, "backend": "mock" if MOCK else BACKEND_URL}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": "dexa-cua-vlm", "object": "model", "owned_by": "dexa"}]}


@app.get("/v1/usage")
async def usage():
    return store.snapshot()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        with open(DASHBOARD) as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard not found</h1>", status_code=404)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    session = request.headers.get("x-dexa-session") or body.get("user") or "default"
    baseline = request.headers.get("x-dexa-baseline", "gpt-4o")
    if baseline not in pricing.PRICES:
        baseline = "gpt-4o"

    messages = body.get("messages", [])
    dims, img_bytes, text = _extract(messages)
    text_in = _tok_estimate(text)

    # redundancy on the current (last) screenshot
    redundant = 1.0
    exact_dup = False
    if img_bytes:
        obs = meter.observe(session, img_bytes[-1])
        redundant, exact_dup = obs.redundant_frac, obs.exact_duplicate

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    img_hash = hashlib.sha256(img_bytes[-1]).hexdigest() if img_bytes else ""
    cache_hit = False

    # exact-duplicate reuse: identical screenshot + identical prompt -> serve cached, skip model
    prev = _cache.get(session)
    if exact_dup and prev and prev[0] == img_hash and prev[1] == text_hash:
        resp = dict(prev[2])
        cache_hit = True
    elif MOCK:
        resp = _mock_completion("dexa-cua-vlm", text)
    else:
        resp = await _forward(body)
    resp["model"] = "dexa-cua-vlm"
    _cache[session] = (img_hash, text_hash, resp)

    out_tokens = (resp.get("usage") or {}).get("completion_tokens") or _tok_estimate(
        (resp["choices"][0]["message"].get("content") or ""))
    saved = pricing.compare(dims, text_in, out_tokens, baseline_model=baseline)
    if cache_hit:  # a served-from-cache step costs us ~nothing
        saved.ours.usd = 0.0
        saved.saved_usd = saved.baseline.usd
        saved.saved_pct = 100.0
        saved.x_cheaper = float("inf")

    store.record(session, saved=saved, redundant_frac=redundant, cache_hit=cache_hit)

    info = saved.as_dict()
    info.update({"session": session, "screen_redundancy_pct": round(100 * redundant, 1),
                 "served_from_cache": cache_hit})
    resp["dexa"] = info

    headers = {
        "x-dexa-cost-usd": f"{saved.ours.usd:.8f}",
        "x-dexa-baseline-usd": f"{saved.baseline.usd:.8f}",
        "x-dexa-saved-usd": f"{saved.saved_usd:.8f}",
        "x-dexa-saved-pct": f"{saved.saved_pct:.1f}",
        "x-dexa-x-cheaper": ("inf" if math.isinf(saved.x_cheaper) else f"{saved.x_cheaper:.1f}"),
        "x-dexa-screen-redundancy-pct": f"{100 * redundant:.1f}",
        "x-dexa-cache": "hit" if cache_hit else "miss",
        "x-dexa-baseline-model": baseline,
    }
    return JSONResponse(resp, headers=headers)
