"""A tiny stand-in for a real vLLM OpenAI backend — no GPU.

Mimics the contract the gateway depends on: POST /v1/chat/completions returning a chat
completion with a `usage` block, and GET /health. Used to test BYOC forwarding end to end
(the gateway forwarding to a *customer's own* backend) without a model.

    uvicorn dexa_platform.tests.stub_vllm:app --port 8091
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "unknown")
    # echo which model the gateway asked for, so the test can assert the rewrite happened
    return {
        "id": "chatcmpl-stub", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": f"STUB_OK model={model}"}}],
        "usage": {"prompt_tokens": 512, "completion_tokens": 7, "total_tokens": 519},
    }
