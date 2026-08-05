# Dexa edge — control plane + stateful sessions on Cloudflare

The global front door for Dexa's **stateful inference sessions**. Cloudflare runs the control
plane and session orchestration; the GPU serving tier (vLLM + LMCache) runs on Modal. This is
where onboarding, auth, session routing, and metering live.

Why Cloudflare + Postgres is the right fit — and not just trendy:

| Concern | Primitive | Why |
|---|---|---|
| **A session** | **Durable Object** (`SessionDO`) | per-session durable state, single-writer turn ordering, and **backend affinity** (a session's turns hit the same GPU replica → its KV stays warm). Sessions map onto DOs almost 1:1. |
| API edge, auth, routing | **Workers** | one global deploy at `api.dexa.dev`, ~0 cold start |
| API-key → tenant cache | **Workers KV** | single-read auth on the hot path; revoke propagates within TTL |
| accounts / usage / billing | **Postgres via Hyperdrive** | source of truth (Neon/Supabase), pooled + accelerated from Workers |
| async metering | **Queues** | hot path emits an event and returns; a consumer folds it into Postgres |
| logs / opt-in archive | **R2** | cheap object storage |

The hot path never blocks on Postgres: keys and credit resolve from KV, usage is enqueued.

## The developer experience: sessions are one optional field

The lowest-friction path — your existing OpenAI call, plus a `session` id:

```python
client = OpenAI(base_url="https://api.dexa.dev/v1", api_key="dexa_live_…")
client.chat.completions.create(
    model="dexa-cua-vlm", messages=[...],
    extra_body={"session": "agent-run-42"})   # 1st call cold; every repeat restores warm
```

First call with a new `session` → cold prefill + materialize. Every repeat with the same id →
the KV is restored (via LMCache) instead of re-prefilled. Nothing else in your agent changes.

Power users who want to pre-warm a big context or manage lifecycle use the explicit API:

```
POST   /v1/sessions            {system?, context?}   -> session_id (optionally prewarmed)
POST   /v1/sessions/{id}/turn  {content}             -> warm restore + savings telemetry
GET    /v1/sessions/{id}                              -> state + accumulated savings
DELETE /v1/sessions/{id}                              -> drop (frees KV)
```

Every turn returns whether the KV was **restored (warm)** or **prefilled (cold)**, the tiering
decision (warm→ram→nvme→drop with break-even rationale from the measured cost model), and the
GPU compute saved vs a stateless re-prefill.

## Onboarding (PLG, no card)

1. **Sign up** with GitHub → org + first key + free credits.
2. **See it work in the playground** — paste a big context, watch turn 2 restore warm with the
   savings meter. Or copy the one-field snippet above.
3. **Point your agent** — you already resend the same big prefix every step; add a `session`
   id and it gets warm. Adapters for LangGraph / browser-use / the OpenAI SDK are one line.
4. **Dashboard** — sessions, warm-restore rate, GPU compute saved, tiering.
5. **Later** — add a card for higher limits; BYOC to run the GPU backend in your own VPC.

## Which workloads to bring (and which not to)

**Bring:** long-context + reused + idle — coding agents (repo in context), document/corpus
agents, support agents with long histories, computer-use/browser agents, multi-agent and
human-in-the-loop flows. These idle past a stateless provider's cache TTL and pay full
re-prefill on resume; sessions restore instead.

**Don't bring:** short stateless completions, high-QPS classification, no context reuse — use
the plain `/v1/chat/completions` pass-through (no `session` field); sessions add nothing there.

## Deploy

```bash
npm install
# 1. Postgres (Neon/Supabase), then wire Hyperdrive:
wrangler hyperdrive create dexa-db --connection-string="postgres://…"   # -> put id in wrangler.toml
psql "$DATABASE_URL" -f schema.sql
# 2. KV + Queue + R2:
wrangler kv namespace create KEYS                # -> put id in wrangler.toml
wrangler queues create dexa-usage
wrangler r2 bucket create dexa-archive
# 3. point GPU_BACKEND_URL (wrangler.toml [vars]) at your Modal vLLM+LMCache deploy
wrangler deploy
```

## Status

Deployable scaffold. The session/tiering/metering logic is ported from the Python service in
`../dexa_platform/` (which is runtime-tested and has a live GPU backend); this Cloudflare port
has **not** been run in CF yet — treat it as the infra starting point. Not yet wired: GitHub
OAuth + Stripe (secrets scaffolded in `wrangler.toml`), the console UI, and driving the NVMe
tier from the tiering policy (today LMCache manages CPU eviction; the policy is advisory).

## Map to the Python service

| edge (TS) | dexa_platform (Python, tested) |
|---|---|
| `src/session.ts` (SessionDO) | `sessions/service.py` + `sessions/store.py` |
| `src/tiering.ts` | `sessions/tiering.py` |
| `src/keys.ts` + KV | `control/resolver.py` |
| `src/db.ts` + Postgres | `control/db.py` + `control/models.py` |
| Queue consumer | `control/metering.py` |
