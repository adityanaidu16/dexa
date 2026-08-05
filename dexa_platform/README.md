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
gateway/app.py          OpenAI-compatible FastAPI gateway (/v1/chat/completions, /v1/usage)
dashboard/index.html    live savings dashboard
examples/switch.py      the two-line drop-in
examples/agent_loop.py  a computer-use agent trajectory on Dexa
tests/                  pricing + redundancy unit tests
```

## Status

MVP: cost accounting, redundancy meter, exact-frame cache, gateway, dashboard, tests — all
runnable. Not yet built: auth/accounts/billing, Redis-backed store, streaming passthrough,
real agent-trajectory (OSWorld/WebArena) validation of the redundancy numbers.
