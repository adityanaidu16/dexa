# Dexa CUA backend — deploy guide

The backend is Qwen2.5-VL-7B served through vLLM's OpenAI-compatible API, tuned for
computer-use agents (prefix caching on, screenshot-appropriate visual-token budget, multiple
images per prompt). It serves model name `dexa-cua-vlm` on port `8000`. The gateway forwards
to it at `$URL/v1/chat/completions`.

The launch flags live in exactly one place — `_vllm_cmd()` in `cua_backend.py` (mirrored by
`backend_launch.sh`) — so hosted and BYOC deployments are byte-for-byte the same server.

## Option A — Modal (Dexa-hosted, or your own Modal account)

```bash
modal deploy serve/cua_backend.py        # stable URL, scales to zero when idle
# -> https://<you>--dexa-cua-backend-serve.modal.run
modal app stop dexa-cua-backend          # tear down
```

Set the gateway's `DEXA_BACKEND_URL` to that URL.

## Option B — BYOC, on your own GPU (data never leaves your network)

Any A100/H100-80GB box in your cloud:

```bash
pip install "vllm==0.24.0" qwen-vl-utils
DEXA_MAX_PIXELS=1050000 ./serve/backend_launch.sh     # serves :8000 as dexa-cua-vlm
```

Or with the official vLLM image via docker-compose (gateway + backend together):

```bash
cd dexa_platform && docker compose -f docker-compose.byoc.yml up
```

Then run the gateway pointed at it:

```bash
DEXA_BACKEND_URL=http://<backend-host>:8000 DEXA_MODE=byoc \
  DEXA_API_KEY=byoc-yourkey uvicorn dexa_platform.gateway.app:app --port 8080
```

## Tuning knobs (env)

| var | default | meaning |
|-----|---------|---------|
| `DEXA_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | any Qwen2.5-VL checkpoint |
| `DEXA_MAX_PIXELS` | `1050000` | visual-token budget per image; lower = cheaper/faster, higher = more legible small UI text |
| `DEXA_MAX_MODEL_LEN` | `16384` | context length (tool defs + history frames) |
| `DEXA_MAX_IMAGES` | `3` | images allowed per prompt (current + history frames) |

`DEXA_MAX_PIXELS` is the main cost/quality dial. The evals in `../evals/RESULTS.md` establish
the accuracy/cost curve; `1050000` (~1.05MP) keeps 1280×800-class screenshots legible while
billing ~300–1300 visual tokens vs ~1,100 (GPT-4o) or ~35,000 (GPT-4o-mini) for the same image.

## Health

vLLM exposes `GET /health` (200 when ready). The gateway surfaces backend config at
`GET /healthz`. Cold start on Modal is ~1–3 min (model load); warm requests are immediate.
