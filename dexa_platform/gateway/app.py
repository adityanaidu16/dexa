"""Dexa gateway — an OpenAI-compatible endpoint for computer-use agents.

Switching cost is two lines: point your OpenAI client's base_url at this server and use your
Dexa key. Every `/v1/chat/completions` call is served by the optimized VLM backend for your
tenant, and the response carries — in the body under `dexa` and in `x-dexa-*` headers — what
the call cost here vs the frontier model you came from, plus the measured screen redundancy.

Deployment shapes (same code):
  * Hosted     — Dexa runs the gateway + backend; you just send requests.
  * BYOC       — you run this gateway and the backend in your own cloud (docker-compose.byoc
                 .yml). Screenshots never leave your network; Dexa is the serving recipe +
                 telemetry. Set DEXA_BACKEND_URL to your backend and DEXA_MODE=byoc.

    uvicorn dexa_platform.gateway.app:app --port 8080

Privacy: the gateway persists no screenshots or completions. The only image bytes held are a
per-session hash of the last frame (for exact-duplicate dedup) plus the last completion, in
process memory; set DEXA_CACHE=0 (or a tenant's cache_enabled=false) to disable even that.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import pricing
from .redundancy import RedundancyMeter
from .store import Store
from .tenants import TenantRegistry

import os
DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")

# Accounts mode: resolve DB-backed API keys, enforce free-credit quota, meter durably.
# Off by default so the static/BYOC self-host + local-demo paths keep working unchanged.
ACCOUNTS = os.environ.get("DEXA_ACCOUNTS") == "1"
if ACCOUNTS:
    from ..control import db as cp_db
    from ..control import metering as cp_metering
    from ..control.resolver import KeyResolver
    cp_db.init()
    resolver = KeyResolver()

app = FastAPI(title="Dexa Gateway", version="0.3.0")
registry = TenantRegistry.load()
meter = RedundancyMeter()
store = Store()
# per internal-session last (image_hash, text_hash) -> cached response, for exact-dup reuse
_cache: dict[str, tuple[str, str, dict]] = {}


def _tok_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # ~4 chars/token; good enough for cost display


_DATA_URI = re.compile(r"^data:image/[^;]+;base64,(.*)$", re.DOTALL)


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def _extract(messages: list) -> tuple[list[tuple[int, int]], list[bytes], str]:
    """Return (image dims, image bytes, concatenated text)."""
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


async def _forward(tenant, body: dict) -> dict:
    payload = dict(body)
    payload["model"] = tenant.backend_model
    payload["stream"] = False
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{tenant.backend_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()


@app.get("/healthz")
async def healthz():
    return {"ok": True, "version": app.version, **registry.summary()}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": "dexa-cua-vlm", "object": "model", "owned_by": "dexa"}]}


@app.get("/v1/usage")
async def usage():
    if ACCOUNTS:
        with cp_db.session() as s:
            return cp_metering.global_snapshot(s)
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
    org_id = key_id = None
    if ACCOUNTS:
        principal = resolver.resolve(_bearer(request))
        if principal is None:
            return JSONResponse(
                {"error": {"message": "invalid or missing API key",
                           "type": "invalid_request_error", "code": "invalid_api_key"}},
                status_code=401)
        with cp_db.session() as s:
            if not cp_metering.quota_ok(s, principal.org_id):
                return JSONResponse(
                    {"error": {"message": "free credit exhausted — add a payment method to "
                               "continue", "type": "insufficient_quota",
                               "code": "insufficient_quota"}}, status_code=402)
        tenant, org_id, key_id = principal.tenant, principal.org_id, principal.key_id
    else:
        tenant = registry.resolve(_bearer(request))
        if tenant is None:
            return JSONResponse(
                {"error": {"message": "invalid or missing API key",
                           "type": "invalid_request_error", "code": "invalid_api_key"}},
                status_code=401)

    body = await request.json()
    raw_session = request.headers.get("x-dexa-session") or body.get("user") or "default"
    session = f"{tenant.name}/{raw_session}"          # isolate tenants in the ledger
    baseline = request.headers.get("x-dexa-baseline", tenant.default_baseline)
    if baseline not in pricing.PRICES:
        baseline = "gpt-4o"

    messages = body.get("messages", [])
    dims, img_bytes, text = _extract(messages)
    text_in = _tok_estimate(text)

    redundant, exact_dup = 1.0, False
    if img_bytes:
        obs = meter.observe(session, img_bytes[-1])
        redundant, exact_dup = obs.redundant_frac, obs.exact_duplicate

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    img_hash = hashlib.sha256(img_bytes[-1]).hexdigest() if img_bytes else ""
    cache_hit = False

    prev = _cache.get(session) if tenant.cache_enabled else None
    try:
        if exact_dup and prev and prev[0] == img_hash and prev[1] == text_hash:
            resp, cache_hit = dict(prev[2]), True
        elif tenant.uses_mock:
            resp = _mock_completion("dexa-cua-vlm", text)
        else:
            resp = await _forward(tenant, body)
    except httpx.HTTPError as e:
        return JSONResponse(
            {"error": {"message": f"backend error: {e}", "type": "backend_error"}},
            status_code=502)

    resp["model"] = "dexa-cua-vlm"
    if tenant.cache_enabled:
        _cache[session] = (img_hash, text_hash, resp)

    out_tokens = (resp.get("usage") or {}).get("completion_tokens") or _tok_estimate(
        (resp["choices"][0]["message"].get("content") or ""))
    saved = pricing.compare(dims, text_in, out_tokens, baseline_model=baseline)
    if cache_hit:
        saved.ours.usd = 0.0
        saved.saved_usd = saved.baseline.usd
        saved.saved_pct = 100.0
        saved.x_cheaper = float("inf")

    if ACCOUNTS:
        with cp_db.session() as s:
            cp_metering.record(
                s, org_id, key_id,
                dexa_usd=saved.ours.usd, baseline_usd=saved.baseline.usd,
                saved_usd=saved.saved_usd, dexa_image_tokens=saved.ours.image_tokens,
                baseline_image_tokens=saved.baseline.image_tokens,
                redundant_frac=redundant, cache_hit=cache_hit)
    else:
        store.record(session, saved=saved, redundant_frac=redundant, cache_hit=cache_hit)

    info = saved.as_dict()
    info.update({"tenant": tenant.name, "mode": tenant.mode, "session": raw_session,
                 "screen_redundancy_pct": round(100 * redundant, 1),
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
        "x-dexa-tenant": tenant.name,
    }
    return JSONResponse(resp, headers=headers)
