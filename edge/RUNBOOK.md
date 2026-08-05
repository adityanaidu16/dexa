# Deploy runbook — take the edge live

Goal: one real request flowing through the whole path — **key auth → Session Durable Object →
Modal GPU backend → warm restore → usage in Postgres** — in ~20 minutes. Prereqs: a Cloudflare
account, `npm i -g wrangler` (then `wrangler login`), and `psql`.

## 1. Postgres (source of truth)

Create a serverless Postgres (Neon or Supabase); grab its connection string as `DATABASE_URL`.

```bash
export DATABASE_URL="postgres://user:pw@host/dexa?sslmode=require"
psql "$DATABASE_URL" -f schema.sql            # creates orgs/users/api_keys/usage_daily/…
```

## 2. Cloudflare resources → fill in wrangler.toml

```bash
wrangler hyperdrive create dexa-db --connection-string="$DATABASE_URL"   # -> [[hyperdrive]] id
wrangler kv namespace create KEYS                                        # -> [[kv_namespaces]] id
wrangler queues create dexa-usage
wrangler r2 bucket create dexa-archive
```

Paste the printed Hyperdrive id and KV id into `wrangler.toml` (replace the `REPLACE_WITH_*`).
Point `GPU_BACKEND_URL` in `[vars]` at your Modal deploy (already set to the current one).

## 3. GPU backend (data plane)

```bash
cd .. && modal deploy serve/vllm_lmcache_backend.py     # -> https://…modal.run
```

Put that URL in `wrangler.toml` `GPU_BACKEND_URL` if different.

## 4. Mint a working key (until OAuth is wired)

```bash
cd edge && node scripts/seed.mjs you@example.com        # prints a dexa_live_… key ONCE
export DEXA_KEY="dexa_live_…"
```

## 5. Deploy the Worker

```bash
npm install
npm run typecheck                                        # tsc --noEmit, should be clean
wrangler deploy                                          # -> https://dexa-edge.<sub>.workers.dev
export DEXA_EDGE_URL="https://dexa-edge.<sub>.workers.dev"
```

## 6. Prove it end-to-end

```bash
node scripts/smoke.mjs
# 1) create session with a big context …
# 2) turn 1 (COLD prefill)   live=…ms
# 3) turn 2 (WARM restore)   live=…ms  tier=warm
# PASS: turn 2 restored warm, ~Nx faster end-to-end than the cold turn 1.
```

Then check usage landed:

```bash
psql "$DATABASE_URL" -c "select org_id, requests, warm_turns, cold_turns, saved_usd from usage_daily;"
```

The one-field DX also works now:

```bash
curl -s $DEXA_EDGE_URL/v1/chat/completions -H "authorization: Bearer $DEXA_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"dexa-cua-vlm","messages":[{"role":"user","content":"hi"}],"session":"run-1"}'
```

## Notes

- First backend request after idle re-warms the Modal container (~cold start); the smoke may
  see a slower turn 1 for that reason on top of the real prefill.
- `wrangler tail` streams live Worker logs while you test.
- Local dev: `wrangler dev` runs Workers + DO + KV + Queues locally (miniflare); Hyperdrive
  needs a reachable Postgres. Use a Neon branch as your dev DB.
