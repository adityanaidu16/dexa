# Dexa — self-serve inference for computer-use agents

An **OpenAI-compatible** endpoint for teams building computer-use agents (or running them in
sandboxes). You move a workload onto Dexa in **two lines**, and from the very first request
you see — in the response and on a live dashboard — exactly what the call cost here versus
what it would have cost on the frontier model you came from.

Two lines is the whole switch:

```python
# before
client = OpenAI(api_key="sk-...")                 ; MODEL = "gpt-4o"
# after
client = OpenAI(base_url="https://api.dexa.dev/v1", api_key="dexa-...") ; MODEL = "dexa-cua-vlm"
```

Same SDK, same `messages`, same `image_url` content. Nothing else in your agent changes.

## Why it's cheaper (measured, not pitched)

Computer-use cost is dominated by how many **visual tokens** a provider bills for each
screenshot, and providers count the same image very differently. Dexa serves an optimized
open VLM (Qwen2.5-VL-7B) at the visual-token budget our evals found is the accuracy/cost
sweet spot. For a 1280×800 screenshot:

| provider | image tokens billed | rel. cost of one step |
|----------|--------------------:|----------------------:|
| GPT-4o-mini | ~35,000 | highest |
| GPT-4o | 1,105 | 1× |
| **Dexa (Qwen2.5-VL)** | **~324** | **~40× cheaper than 4o** |

On our DocVQA benchmark the same open model matched GPT-4o's accuracy (0.925 vs 0.880) at
this budget — so the cost cut isn't bought with quality. See `../evals/RESULTS.md`.

## What you feel immediately

Every `/v1/chat/completions` response carries the impact, so you don't have to instrument
anything:

- **In the body**, under `dexa`: `dexa_cost_usd`, `baseline_cost_usd`, `x_cheaper`,
  `saved_pct`, `screen_redundancy_pct`, `served_from_cache`.
- **In headers**: `x-dexa-cost-usd`, `x-dexa-baseline-usd`, `x-dexa-saved-usd`,
  `x-dexa-saved-pct`, `x-dexa-x-cheaper`, `x-dexa-screen-redundancy-pct`, `x-dexa-cache`.
- **On a live dashboard** at `/dashboard` — cumulative savings, ×cheaper, per-session table.

Set `x-dexa-session` to group a trajectory and `x-dexa-baseline` (`gpt-4o` / `gpt-4o-mini` /
`claude-3-5-sonnet`) to price against whatever you're switching from.

## Deployment shapes: hosted or BYOC

The same gateway runs two ways, selected per API key (a *tenant*):

- **Hosted** — Dexa runs the gateway + backend; you send requests. Fastest to try.
- **BYOC** — you run this gateway *and* the backend in your own cloud/VPC, so screenshots
  never leave your network. Dexa is the serving recipe + savings telemetry, not a data path.

BYOC in one command (gateway + real backend together, needs an 80GB GPU host):

```bash
cd dexa_platform && DEXA_API_KEY=byoc-yourkey docker compose -f docker-compose.byoc.yml up
# gateway on :8080 -> backend on :8000, both in your network
```

Or run the gateway pointed at a backend you already operate:

```bash
DEXA_BACKEND_URL=http://your-vllm:8000 DEXA_MODE=byoc \
  DEXA_API_KEY=byoc-yourkey DEXA_REQUIRE_AUTH=1 \
  uvicorn dexa_platform.gateway.app:app --port 8080
```

Multi-tenant hosted control plane: put tenants in a JSON file (see
`config/tenants.example.json`) and set `DEXA_TENANTS=/path/to/tenants.json`. Auth is a Bearer
API key; unknown keys get `401`. `GET /healthz` shows tenant config with **no keys** exposed.

**Data handling.** The gateway persists no screenshots or completions. The only image bytes
held are a per-session hash of the last frame (for exact-duplicate dedup) plus the last
completion, in process memory. Set `DEXA_CACHE=0` (or a tenant's `cache_enabled=false`) to
disable even that.

## Accounts, keys & metering (self-serve / PLG)

The control plane (`control/`) turns the gateway into something people can sign up for. It's
SQLite in dev, Postgres in prod — one env var (`DATABASE_URL`) changes it.

- **PLG sign-up** (`POST /v1/signup`): an OAuth identity in → an org, a user, a first API key,
  and a **free credit grant** out. No card. Idempotent — the key secret is shown exactly once.
- **API keys**: `dexa_live_…` secrets, SHA-256 hashed at rest (only a short prefix is stored
  for display). Create / list / rotate / revoke from the control API.
- **Durable metering**: every request upserts a per-key daily rollup and debits the org's
  credit. When free credit runs out, the gateway returns `402` until a card is added.
- **Hot-path safe**: the gateway resolves keys from a short-TTL cache (Redis-swappable), so
  auth doesn't hit Postgres per request; a revoke propagates within the TTL.

Run both planes locally (shared SQLite):

```bash
# control plane (console/CLI call this) on :8090
DATABASE_URL=sqlite:///./dexa.db uvicorn dexa_platform.control.api:app --port 8090
# data plane in accounts mode on :8080
DATABASE_URL=sqlite:///./dexa.db DEXA_ACCOUNTS=1 ./dexa_platform/run_local.sh
```

```bash
# self-serve in three calls:
KEY=$(curl -s :8090/v1/signup -d '{"provider":"github","subject":"me","email":"me@co.com"}' \
      -H 'content-type: application/json' | jq -r .api_key)
curl :8080/v1/chat/completions -H "authorization: Bearer $KEY" -H 'content-type: application/json' -d @req.json
curl :8090/v1/me -H "authorization: Bearer $KEY"      # remaining credit + usage
```

Accounts mode is **off by default** (`DEXA_ACCOUNTS` unset) so the static/BYOC self-host and
local-demo paths keep working with no database.

Control-plane env: `DATABASE_URL` (dev SQLite / prod Postgres), `DEXA_FREE_CREDIT_USD`
(default 5), `DEXA_KEY_CACHE_TTL` (default 30s).

## Stateful sessions (the differentiated product)

The thing a stateless provider structurally can't do: a **durable, warm session**. Prefill a
big context once; on later turns its KV is **restored** (from GPU/CPU/NVMe via LMCache) instead
of re-prefilled. Measured advantage: 11–34× faster resume, growing with context, and 2–6×
cheaper per long-context session — see `../docs/STATEFUL_SESSIONS.md` and `../evals/RESULTS.md`.

```
POST   /v1/sessions            create a session (optionally with a big initial context)
POST   /v1/sessions/{id}/turn  run a turn — KV restored (warm), not re-prefilled
GET    /v1/sessions/{id}        state + accumulated savings
DELETE /v1/sessions/{id}        drop it (frees the KV)
```

Each turn returns whether the KV was **restored (warm)** or **prefilled (cold)**, the tiering
decision (warm→ram→nvme→drop, with break-even rationale from the cost model), and the modeled
GPU compute saved vs a stateless re-prefill.

```bash
# real backend (vLLM + LMCache) on Modal:
modal deploy serve/vllm_lmcache_backend.py
DEXA_SESSION_BACKEND=https://<url> uvicorn dexa_platform.sessions.service:app --port 8070
# or GPU-free, simulated from the measured profile:
DEXA_SESSION_MOCK=1 uvicorn dexa_platform.sessions.service:app --port 8070
```

Verified live: a 12k-token session cold-prefills in ~5.4 s (turn 1), then restores warm in
~1.5 s (turn 2) end-to-end through the deployed vLLM+LMCache backend — 3.7× faster turn-over-
turn (the isolated in-engine restore win is ~11×; HTTP + decode overhead compresses the
end-to-end ratio). `tiering.py` encodes the warm/RAM/NVMe break-evens from the cost model.

## The backend

The model server the gateway forwards to is `../serve/cua_backend.py` — Qwen2.5-VL-7B tuned
for computer-use agents (prefix caching on for the static system/tool prefix agents resend
every step, a screenshot-appropriate `max_pixels` budget, multiple images per prompt). The
launch flags live in one place and are shared byte-for-byte by the Modal deploy and the BYOC
Docker/CLI launch, so hosted and self-hosted cost/quality are identical. Deploy guide:
`../serve/BACKEND.md`.

## Redundancy meter + exact-frame cache (honest scope)

Agents re-send near-identical screenshots every step. The gateway measures, per session, how
much of the screen actually changed (`screen_redundancy_pct`) — the honest headroom number
(our evals measured ~86% redundant screens). Today it **acts** on the safe part of that: when
a screenshot is byte-identical to the previous one and the prompt matches, the completion is
served from cache — a true 100%-saved step, zero accuracy risk. The larger
"reuse-compute-for-the-changed-but-stable-frame" prize is real but gated on delta-encoding
R&D our evals showed caps near ~2×, so we report it as headroom and don't bill it as magic.

## Run it locally (no GPU needed)

```bash
pip install -r requirements.txt openai
./run_local.sh                                   # mock backend — for the demo + dashboard
python examples/agent_loop.py --steps 20         # drive a trajectory
open http://localhost:8080/dashboard             # watch savings accrue
```

Point at the real backend (the Modal Qwen2.5-VL endpoint in `../serve/`):

```bash
DEXA_BACKEND_URL=https://<your-modal-url> ./run_local.sh
```

## Layout

```
gateway/pricing.py      per-provider image→token accounting + cost comparison (the impact core)
gateway/redundancy.py   per-session frame redundancy + exact-duplicate detection
gateway/store.py        in-memory usage/savings ledger (swap for Redis/Postgres for accounts)
gateway/tenants.py      API-key -> tenant registry; hosted / BYOC / mock routing
gateway/app.py          OpenAI-compatible FastAPI gateway (auth, routing, telemetry)
config/tenants.example.json   multi-tenant config sample
dashboard/index.html    live savings dashboard
Dockerfile              gateway image (CPU-only)
docker-compose.byoc.yml gateway + backend in your own cloud
examples/switch.py      the two-line drop-in
examples/agent_loop.py  a computer-use agent trajectory on Dexa
tests/                  pricing, redundancy, tenant tests + a vLLM-API stub for forwarding tests
../serve/cua_backend.py the real Qwen2.5-VL CUA backend (Modal + BYOC); ../serve/BACKEND.md
```

## Status

Runnable now: cost accounting, redundancy meter, exact-frame cache, multi-tenant gateway with
API-key auth, hosted **and** BYOC routing, the real CUA-tuned backend (Modal deploy + BYOC
Docker/CLI), live dashboard, 15 passing tests incl. a GPU-free BYOC forwarding stub. Not yet
built: billing/quotas, Redis-backed store, streaming passthrough, and real agent-trajectory
(OSWorld/WebArena) validation of the redundancy numbers.
