## 1. Thesis

Candidate H is an OpenAI-compatible inference endpoint for screenshot-heavy computer-use agents (CUA): an open VLM served at a screenshot-sized visual-token budget, a gateway that dedups exact-repeat frames, pins a session's static system/tool prefix to a warm replica, and returns per-request "what this step cost here vs. the frontier model you left" telemetry, offered hosted or as a bring-your-own-cloud (BYOC) recipe. The buyer is a team running a browser/desktop agent loop (Browser Use, Skyvern, OpenHands-style harnesses, QA/RPA agents) that pays per screenshot today. The sentence the customer would repeat: "We pointed our agent at Dexa with a base_url change, every step shows the saving, and the screenshots never left our VPC." Honest state of the evidence: the visual-token arithmetic is reproducible (322 vs 1,105 tokens per 1280x800 frame); the "~40x cheaper" headline is a price model, not a measurement; the exact-frame cache and prefix warmth are shipped but unvalidated on real trajectories; the delta-perception prize is bounded at ~2.3x. The gateway and control plane exist in `dexa_platform/`; the backend is stock vLLM with tuned flags.

## 2. Customer and workload

Who buys: engineering teams shipping agents that act on a screen. Demand-side shape from the research files: Browser Use (112.1k GitHub stars) ships a hosted model plus an open-weight `bu-30b-a3b-preview` (30B total / 3B active MoE, 2025-12-16) it claims runs "200 tasks per $1", and supports OpenAI, Anthropic, Gemini and custom providers; Skyvern (22.9k stars) accepts any OpenAI-compatible endpoint; OpenAI folded Operator into ChatGPT Agent and the Agents SDK on 2025-08-31; Cursor's Cloud Agents added computer use in February 2026 (medium confidence) [05-research-agents.md]. Morph lists Glance (`morph-computer-use-v0`, "AI testing, 10x cheaper than general-purpose"), no fetched pricing or benchmark [05-research-morph.md].

What they run today: one screenshot per step plus a static prefix and a short action. On Anthropic's GA computer-use toolset a screenshot costs roughly 1,000-1,800 input tokens, the toolset adds ~4,500 tokens (browser toolset ~6,600), and the docs advise 20 or fewer images per request and pruning screenshots in batches (keep the last three, prune every 25 turns) so the cached prefix stays byte-identical [05-research-agents.md]. Claude bills ceil(w/28) x ceil(h/28) tokens per image (1000x1000 = 1,296; standard tier cap 1,568) [05-research-docai.md], so a 1280x800 frame is 46 x 29 = 1,334 tokens (derived). GPT-4o bills 85 + 170 per 512px tile: 1,105 for 1280x800 [dexa_platform/gateway/pricing.py]. Qwen2.5-VL at native resolution bills 322 for the same frame (`pricing.qwen_vision_tokens`, run this session; README says ~324).

Turn, idle and concurrency pattern: unmeasured for CUA anywhere in the repos or research files. Nearest analogs are coding agents: Mooncake's Codex traces show 5.2 s median / 81.4 s p99 tool-driven gaps; TraceLab shows a 1.4 min median human gap (p90 20.6 min) and 95.7% prefix-cache hits [05-research-agents.md]. The dexa redundancy bench waited 180 ms between actions, a harness setting, not a workload measurement [evals/agent_redundancy/bench.py].

How they pay today: per token. Sonnet 5 $2/$10 per MTok, Opus 5 $5/$25, Haiku 4.5 $1/$5, cache reads 0.1x with a 5-minute default TTL (1-hour at 2x write) [05-research-caching.md]; gpt-5.4 $2.50/$15, gpt-4o $2.50/$10 [05-research-docai.md]. Hosted open VLMs: ~$0.20/MTok for Qwen3-VL-8B on DeepInfra and Fireworks (secondary aggregator), $0.50/$1.50 for Qwen3-VL-32B on Together [05-research-docai.md].

## 3. The pain, in the customer's words

No customer interviews exist in either repo; the following are the complaints implied by documented provider behavior, written as a buyer would say them (unmeasured as quotes):

- "Every step re-bills a full screenshot, and on Claude that is 1,000-1,800 tokens before I've said a word." (Anthropic computer-use docs.)
- "My cache only works if I keep the prefix byte-identical, so I'm hand-pruning screenshots every 25 turns to keep the hit rate up." (Anthropic caching guidance.)
- "On serverless providers the cache lives in one replica; if I don't send a session-affinity header I pay full price again." (Fireworks: caching "only works within 1 replica"; OpenAI: `prompt_cache_key` influences routing but does not pin; xAI: entries "can be evicted at any time.") [05-research-caching.md; 05-research-agents.md]
- "I can't put customer screens through a third-party API; I need the model in my VPC." (Skyvern and Browser Use support self-hosted/OpenAI-compatible backends; Bland sells on-prem to voice buyers for the same reason.) [05-research-agents.md; 05-research-voice.md]
- "I don't know what a step costs until the invoice." (No provider fetched returns a per-request cost-vs-alternative figure.)

## 4. Value proposition and the proof-of-value benchmark

Value proposition at the strength the ledgers support: (a) fewer billed visual tokens per screenshot than frontier tokenizers (322 vs 1,105 GPT-4o vs 1,334 Claude-formula for 1280x800: 3.4x and 4.1x fewer, derived); (b) a lower per-token price because the model is an open 7B-8B VLM (`pricing.py` assumes $0.20/MTok blended, matching the aggregator price for hosted Qwen3-VL-8B); (c) session affinity so the static prefix hits vLLM's prefix cache (measured 12x TTFT at 4k, 25-34x at 16k, text model, single replica); (d) $0 exact-duplicate steps; (e) per-request savings telemetry; (f) BYOC.

What must be confronted before the benchmark is designed:

1. "~40x cheaper than GPT-4o" is modeled: 1,105/322 = 3.4x fewer image tokens times $2.50/$0.20 = 12.5x assumed price ratio, ~43x on image tokens alone [04-conflicts.md #10]. `pricing.compare` on a full request (one 1280x800 frame, 200 text-in, 20 out) gives 31.9x vs gpt-4o and 46.1x vs claude-3-5-sonnet (pricing.py's Anthropic formula, 1,366 tokens). No measured serving cost per step exists.
2. The accuracy evidence is not on screenshots. Qwen2.5-VL-7B scored 0.925 vs GPT-4o 0.880 on 200 DocVQA pages (relaxed match) at the 1024px budget, which is 1,017 visual tokens per page, not 322 [evals/RESULTS.md 'Multimodal execution thesis'; 04-conflicts.md #10]. Grounding accuracy at 322 tokens is unmeasured.
3. The budget is not a differentiator at 1280x800: `qwen_vision_tokens(1280, 800)` returns 322 with the 1.05 MP cap or with no cap (run this session). The cap bites only above ~1 MP: 1920x1080 is 646 uncapped vs 312 capped; 2560x1440 is 1,196 vs 312. Any Qwen host bills the same 322 tokens for a 1280x800 frame; the token advantage in the README table is the Qwen tokenizer's, not Dexa's.
4. Redundancy prize 7.3x, realized ceiling ~2.3x. Screens are 86.3% unchanged per action (13.7% of 28px patches change; 2.5-6% type/edit/dropdown, ~28% select-row, 22-46% navigation) on a synthetic 15-step Playwright CRM task [evals/RESULTS.md 'Computer-use agent screen redundancy']. But the Qwen2.5-VL encoder spills: 15.6% pixel change moves 52.8% of vision tokens (cos < 0.98), a 3.39x spill; moved tokens' median cosine is 0.83; the ceiling is 2.34x at cos >= 0.98, ~3.1x at >= 0.95, ~3.9x at >= 0.90 (n = 8 frames) [evals/RESULTS.md 'Delta-perception viability', 'Tolerance sweep']. The 7.3x is superseded [04-conflicts.md superseded list].
5. No OSWorld/WebArena validation: `dexa_platform/README.md` 'Status' lists it as not built.
6. The "CUA-tuned backend" is `vllm serve` with flags: `--enable-prefix-caching`, `--max-model-len 16384`, `--gpu-memory-utilization 0.92`, `--limit-mm-per-prompt {"image": 3}`, `--mm-processor-kwargs {"max_pixels": 1050000}`, vLLM 0.24.0, Qwen2.5-VL-7B-Instruct, A100-80GB [serve/cua_backend.py]. No custom serving code.

The proof-of-value benchmark, pre-registered:

- Metric: dollars per successfully completed task and p50 step latency at matched success, on a public CUA suite (OSWorld-Verified and/or a 100-200-task WebArena subset; Browser Use's Odysseys, where it reports 87.4%, as a third). Billed tokens read from vLLM `usage` fields (including cached tokens), not `pricing.py` estimates.
- Setup: one harness (Browser Use or Skyvern, OpenAI-compatible mode), same tasks and seeds, five backends.
- Baselines: (1) vanilla vLLM 0.24.0, same model, default flags (no `max_pixels` cap); (2) SGLang, same model; (3) hosted Qwen3-VL-8B at $0.20/MTok; (4) Anthropic Sonnet 5 with the computer-use toolset and documented caching; (5) Browser Use `bu-30b-a3b-preview` on vLLM.
- Target: unmeasured today. Proceed line: success within 3 points of the same model at native resolution and cost per completed task at most one-third of Sonnet 5 at equal or better success. Kill line: success more than 8 points below baseline (3), or cost per task within 1.5x of baseline (3), because then the product is a Qwen host with telemetry.
- Why a skeptical buyer believes it: harness, trajectories and per-request usage are published; the comparison uses the buyer's own frames via `x-dexa-baseline`; cache and prefix hits are observed `usage` counts, not modeled.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| `dexa_platform/gateway/app.py` (261 lines, FastAPI, v0.3.0) | custom, runnable-demo | OpenAI-compatible `/v1/chat/completions`, Bearer key -> tenant, forwards to backend, exact-dup cache, telemetry headers/body; forces `stream=False` (line 127) |
| `gateway/pricing.py` (149 lines) | custom, 5 tests | Per-provider image-token formulas (OpenAI tiles, GPT-4o-mini, Anthropic w*h/750, Qwen smart-resize) and cost comparison at list prices; Dexa rate is an assumed $0.20/$0.20 |
| `gateway/redundancy.py` (94 lines) | custom, 5 tests | Per-session 28px-patch luma diff (threshold 6); acts only on byte-identical frames |
| `gateway/tenants.py`, `store.py` | custom | hosted / byoc / mock routing per key; in-process usage ledger |
| `dexa_platform/control/*` (620 lines, 7 tests) | custom | PLG signup, SHA-256-hashed keys, daily usage rollups, $5 credit ledger, 402 on exhaustion; no OAuth callback, no Stripe |
| `serve/cua_backend.py`, `backend_launch.sh`, `docker-compose.byoc.yml` | OSS vLLM 0.24.0 + Qwen2.5-VL-7B, custom flags | The model server, identical flags hosted and BYOC |
| Session-affinity router | not built | Map `x-dexa-session` to a replica so the static prefix stays in that replica's prefix cache (single `GPU_BACKEND_URL` today) |
| Streaming passthrough, rate limits, metrics | not built | Listed gaps [03-build-inventory.md] |
| Delta-perception encoder reuse | not built; blocked | Reusing unchanged-region vision embeddings requires reimplementing Qwen2.5-VL mRoPE position handling ([3,1671] vs [3,881]) [evals/RESULTS.md 'Moat test'] |

Where differentiation lives: today, entirely in the gateway (dedup, telemetry, tenant/BYOC routing) and operational choices (model, budget, affinity). Nothing in the scheduler, connector, kernel or model is custom. The one execution-owned lever the repo identified (in-forward-pass visual-token reuse) is bounded at ~2.3x and blocked on mRoPE. vLLM has no image-token pruning flag (RFC #45098 open since 2026-06-10; only video EVS shipped); SGLang's EVS is video-only and incompatible with Qwen2.5-VL positional embeddings [05-research-docai.md].

```
 agent harness (Browser Use / Skyvern / custom)
      | OpenAI SDK, base_url -> Dexa, x-dexa-session, x-dexa-baseline
      v
 +--------------------- gateway (custom) ----------------------+
 | key -> tenant (hosted | byoc | mock) | credit gate (opt.)   |
 | RedundancyMeter: 28px patch diff, sha256(frame)             |
 | exact-dup + same text?  -> cached completion, $0            |
 | else -> [session-affinity router: NOT BUILT] -> replica     |
 | pricing.compare(frame dims, text, out) -> x-dexa-* headers  |
 +--------------------------------------------------------------+
      v                                     v
 hosted: Modal A100-80GB              BYOC: customer VPC, same flags
 vllm serve Qwen2.5-VL-7B --enable-prefix-caching --max-model-len 16384
   --limit-mm-per-prompt {"image":3} --mm-processor-kwargs {"max_pixels":1050000}
```

## 6. Evidence

### Proven

- Image-token accounting, 1280x800: Qwen2.5-VL 322 (README ~324), GPT-4o 1,105, GPT-4o-mini 36,835 (README ~35,000); deterministic, 5 passing tests [dexa_platform/gateway/pricing.py; 01-evidence-ledger-dexa.md gpt4o-docvqa-head-to-head].
- Screen redundancy, synthetic 15-step Playwright CRM task at 1280x800: 13.7% of 28px patches change per action; type/edit/dropdown 2.5-6%, select-row ~28%, navigation 22-46% [evals/RESULTS.md; ledger cua-screen-redundancy].
- Budget vs accuracy on documents (Qwen2.5-VL-7B, vLLM 0.24.0, A100, 200 DocVQA pages, relaxed match): 1536px 0.935 / 2,201 tokens / 11.4 img/s; 1024px 0.925 / 1,017 / 30.2 img/s (2.7x, -1.0 pt); 768px 0.885 (-5.0); 512px 0.735 (-20); 384px 0.445 (-49) [evals/RESULTS.md; ledger doc-vlm-frontier].
- vLLM prefix-cache hit vs cold TTFT (Qwen2.5-7B text, vLLM 0.24.0, A100): 4k ~280 -> ~24 ms (~12x); 16k ~1,250 -> ~45 ms (25-34x over 3 runs) [ledger vllm-warmstart-prefix-cache]; the repo labels this "table stakes" (1.00-1.02x vs a prefix-cached SGLang baseline) [ledger agentic-serving-value-v1].
- Build state: 29 dexa_platform tests pass (pricing 5, redundancy 5, tenants 5, accounts 7, sessions 7); gateway/control/serve landed 2026-08-05 [03-build-inventory.md; git log].

### Bounded / contradicted

- Delta-perception reuse: 15.6% pixel change -> 52.8% of vision tokens shifted (3.39x spill); small actions 15-44% shifted, navigation 93-95%; naive reuse ~2x, not 7.3x [ledger delta-perception-viability]. Tolerance sweep: moved-token median cosine 0.83; ceiling 2.34x at >= 0.98, ~3.1x at >= 0.95, ~3.9x at >= 0.90; bands 24/15/14/27/20% (0.95-0.98 / 0.90-0.95 / 0.80-0.90 / 0.50-0.80 / < 0.50); n = 8 frames; grounding-accuracy tolerance never tested [ledger delta-tolerance-sweep]. Contradiction: `evals/RESULTS.md` 'Why incumbents can't capture it (the moat)' claims a 7.3x prize; the same file's later sections cap it at ~2.3x, and 04-conflicts.md lists the 7.3x as superseded.
- DocVQA parity vs cost: accuracy 0.925 vs 0.880 is measured; "~40x cheaper" is (3.4x tokens) x (12.5x assumed price); the Qwen cost row in `modal_incumbent_docvqa.py` is hardcoded ("$0.02-0.06 / 1k pages, rented A100 serving"); GPT-4o-mini accuracy and measured $/1k pages were never recorded [04-conflicts.md #10]. Contradiction: `dexa_platform/README.md` presents 40x under "measured, not pitched"; FINDINGS.md calls the same result "a real economics/config win, not a novel-kernel moat. Copyable."
- Exact-frame cache: fires only when the frame bytes and the concatenated text both match the previous request in the session (`app.py` keys on `(img_hash, text_hash)`); in a real loop the message history grows each step, so hits require a literal re-send. The only trajectory that exercises it is `examples/agent_loop.py`, which fabricates an identical frame every third step. Hit rate on real trajectories: unmeasured.
- Content-aware pruning (the execution-owned moat): blocked by Qwen2.5-VL mRoPE grid coupling; no accuracy numbers produced [ledger vlm-moat-mrope].

### Unproven

| claim | experiment that proves or kills it | est. cost |
|---|---|---|
| Task success holds at the 322-token screenshot budget | OSWorld-Verified/WebArena 100-200 tasks, same model at native vs capped resolution, 5 seeds | ~40 GPU-h (A100 at Modal $2.50/h ≈ $100) + API spend for the Sonnet 5 arm; 5 days |
| Cost per completed task beats hosted Qwen3-VL-8B and Sonnet 5 | Same run; bill from `usage` fields; publish $/task | included above |
| Exact-dup hit rate on real trajectories is material | Replay recorded OSWorld/Browser Use trajectories through the gateway; count `served_from_cache` | ~2 GPU-h; 2 days |
| Prefix-cache hit rate on CUA prompts (images in history) with and without session affinity across >= 4 replicas | Multi-replica vLLM + a consistent-hash router; measure `usage.prompt_tokens_details.cached_tokens` and TTFT | ~30 GPU-h; 5 days |
| Steps per GPU-hour at the CUA budget (throughput, hence real $/step) | Load test with the voice-inference harness pattern (N-slot loadgen, p95 TTFT gate) on A100 and H100 | ~20 GPU-h; 3 days |
| Delta-perception reuse at cos >= 0.98 preserves grounding accuracy | Grounding eval (ScreenSpot-style clicks) with reused vs fresh embeddings; requires an mRoPE-aware forward pass | ~60 GPU-h + 2-3 engineer-weeks |
| BYOC compose runs on a customer GPU without Dexa involvement | Run `docker-compose.byoc.yml` on a rented A100/H100 box | ~4 GPU-h; 1 day |

## 7. MVP and 6-week build plan

What ships first: the hosted endpoint plus the BYOC compose, with observed (not modeled) telemetry, streaming, session affinity, and a public benchmark page. Reused: `dexa_platform/gateway/{app,pricing,redundancy,tenants,store}.py`, `dexa_platform/control/*`, `serve/cua_backend.py`, `serve/backend_launch.sh`, `dexa_platform/docker-compose.byoc.yml`, `evals/agent_redundancy/*`, `evals/modal_vlm_frontier.py` (budget sweep), `evals/modal_incumbent_docvqa.py` (frontier head-to-head); from voice-inference, `vkv/loadgen/runner.py`, `vkv/metrics/{events,manifest,analyze}.py` and the per-point Modal sweep driver in `evals/modal_arm_a.py` for concurrency-knee measurement.

- Week 1: Trajectory corpus and harness. Wire Browser Use and Skyvern (OpenAI-compatible mode) to the gateway; record 100-200 OSWorld-Verified/WebArena tasks; run the five section-4 baselines on the accuracy axis. New: task runner. Input to the model decision (section 11).
- Week 2: Kill gates 1 (task success at budget) and 3 (exact-dup hit rate on real trajectories). Replace `pricing.py`'s len/4 text estimate and modeled Dexa cost with vLLM `usage` fields, including cached tokens.
- Week 3: Session-affinity router (consistent hash on `x-dexa-session`, least-loaded fallback) across >= 4 vLLM replicas; measure prefix-cache hit rate with and without affinity. SSE streaming passthrough (removes `stream=False`).
- Week 4: Throughput and $/step on A100 and H100 with the voice harness pattern; publish steps/GPU-hour. Harden BYOC compose; add `/metrics`.
- Week 5: Public benchmark page with the pre-registered numbers, trajectories and harness; savings computed from observed tokens against the buyer's chosen baseline price table.
- Week 6: Two design-partner pilots (one hosted, one BYOC); kill-gate review; decide whether the delta-perception R&D (ceiling ~2.3x) is funded.

Out of scope for six weeks: Stripe/invoicing, OAuth, Cloudflare edge, delta-perception engine.

## 8. Pricing model

Options the architecture makes expressible:

1. Per step (per screenshot) flat price. Expressible because visual tokens per frame are deterministic given the budget (322 for 1280x800; 312 for 1080p under the cap), so a step is a bounded unit; a per-token provider cannot quote it without knowing the buyer's screen size.
2. Per token at a blended open-VLM rate (`pricing.py` assumes $0.20/$0.20 per MTok; aggregators list hosted Qwen3-VL-8B at $0.20/MTok). Least differentiated.
3. Cached steps at $0 and prefix-hit tokens discounted. Precedents: Tensormesh bills cached input at $0; Fireworks discounts cached tokens 50%; Anthropic reads at 0.1x; Morph and Fireworks use `x-session-id` / `x-session-affinity` headers for sticky placement [05-research-kvcache.md; 05-research-caching.md; 05-research-morph.md]. Expressible only after the affinity router exists.
4. BYOC: recipe-plus-telemetry license; the customer pays their GPU (Modal A100 $2.50/h, H100 $3.95/h; RunPod A100 $1.39-1.59/h; Lambda H100 $3.99-4.29/h [05-research-gpu-pricing.md]).

Cost side: unmeasured for screenshots. The only throughput point is 30.2 document pages/s at 1,017 tokens on one A100, which the repo turned into a hardcoded "$0.02-0.06 / 1k pages" (its cost model assumes $1.80/GPU-h) [evals/RESULTS.md; evals/stateful_cost_model.py:20]. No price can be set until the Week 4 steps/GPU-hour number exists.

## 9. Competitive facts

| who | adjacent thing they ship | what the research files do not show them shipping | source |
|---|---|---|---|
| Anthropic | GA computer-use toolset (17 tools), screenshots ~1,000-1,800 tokens, caching with 5-min/1-h TTL, documented screenshot-pruning pattern | A per-screenshot flat price; a self-hosted/BYOC option; per-request cost-vs-alternative telemetry | 05-research-agents.md; 05-research-caching.md |
| OpenAI | CUA in ChatGPT Agent and Agents SDK; patch-based image tokens (1024x1024 = 1,229 on gpt-5.4); `prompt_cache_key` routing | Machine-pinned caching (docs say it does not pin); BYOC | 05-research-agents.md; 05-research-docai.md; 05-research-caching.md |
| Google Gemini 3 | `media_resolution` low/medium/high = 280/560/1,120 tokens per image | CUA-specific endpoint or per-step billing | 05-research-docai.md |
| DeepSeek | `deepseek-v4-flash-vision-exp` (2026-08-21): images up to 384 tokens each at V4-Flash prices ($0.44/MTok miss, $0.014 hit, peak) | Self-hosting; session affinity guarantees | 05-research-caching.md |
| Browser Use | Own hosted ChatBrowserUse model; open `bu-30b-a3b-preview`; "200 tasks per $1"; 87.4% on Odysseys | An inference API for third-party agents; BYOC serving recipe | 05-research-agents.md |
| Morph | Glance (`morph-computer-use-v0`, "10x cheaper than general-purpose"); sticky KV placement via `x-session-id`; prefix caching on all models | Fetched pricing or benchmark for Glance; BYOC | 05-research-morph.md |
| Fireworks / DeepInfra / Together | Hosted Qwen3-VL (8B at ~$0.20/MTok per aggregators; 32B $0.50/$1.50); Fireworks `x-session-affinity`, 50% cached discount | Screenshot-specific budgets; per-step telemetry; BYOC | 05-research-docai.md; 05-research-agents.md |
| Tensormesh | Cached input billed at $0; Qwen3 30B $0.15/$0.60 | VLM/CUA focus; on-prem "Operator" is "Coming Soon" | 05-research-kvcache.md |
| Skyvern / OpenHands | Any OpenAI-compatible or local endpoint accepted | Their own inference | 05-research-agents.md |
| vLLM / SGLang | Prefix caching; video-token pruning (EVS); open RFC #45098 for image pruning | Shipped image-token pruning; per-session affinity inside the engine (llm-d/SGLang RFCs are external or open) | 05-research-docai.md; 05-research-kvcache.md |

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| Screenshot budget breaks grounding | OSWorld/WebArena success, same model, native vs 1.05 MP cap, 5 seeds | > 8 pts absolute drop | <= 3 pts drop |
| Product is a Qwen host with telemetry | $/completed task vs hosted Qwen3-VL-8B at $0.20/MTok | within 1.5x | <= 1/3 of Sonnet 5 and <= 0.7x of hosted Qwen at equal success |
| Exact-dup cache never fires on real loops | `served_from_cache` rate on recorded trajectories | < 3% of steps | >= 10% of steps |
| Affinity does not move prefix hits | cached_tokens share with vs without router, >= 4 replicas | < 10 pts gain | >= 30 pts gain |
| $/step economics | steps per GPU-hour at p95 step latency <= 2 s on A100/H100 | implied cost > $0.20/MTok-equivalent | implied cost <= 0.5x hosted Qwen price at 50% utilization |
| Delta-perception R&D | grounding accuracy with cos >= 0.98 reuse | > 2 pts drop or reuse < 1.5x | <= 1 pt drop at >= 2x (ceiling is 2.34x) |
| Frontier providers cut visual tokens | Gemini 3 low = 280 tokens; DeepSeek 384; Claude high-res tier | (founder decision, not a measurement) | |

Thresholds are pre-registered choices, not measured values.

## 11. Founder decisions

- Model: Qwen2.5-VL-7B (deployed; 0.925 DocVQA at 1024px) vs Qwen3-VL-8B (DocVQA 96.1 per its report, Apache-2.0, vLLM/SGLang) vs Qwen3-VL-30B-A3B vs a CUA-trained model like Browser Use's 30B-A3B. Evidence: the Week 1 task-success run per model; no CUA-suite number exists in the research files for any of them.
- Hosted vs BYOC vs OSS gateway. Evidence: pilot conversion; BYOC removes the data-residency objection but leaves telemetry as the only paid artifact.
- GPU class: A100 ($1.39-2.79/h across providers), H100 ($1.73-12.29/h), L40S ($0.99-1.95/h), MI300X ($2.59-6.00/h) [05-research-gpu-pricing.md]. Evidence: Week 4 steps/GPU-hour per class.
- Whether frontier adjacency matters: Gemini 3's 280-token low-res mode, DeepSeek's 384-token images at $0.44/MTok, Anthropic's toolset and caching guidance, Morph's Glance. Facts stated neutrally above; the judgment is the founder's.
- Whether to fund the delta-perception engine given the ~2.3x ceiling and the mRoPE wall, or treat ~2x as sufficient.
- Pricing unit: per step vs per token vs $0-cached; publish the baseline price table or let buyers set it (`x-dexa-baseline`).
- Benchmark suite and success definition (OSWorld-Verified, WebArena, Odysseys) and the seed count that counts as proof.
- Whether the "cheaper-VLM reseller" framing FINDINGS.md rejected is acceptable as a wedge. Market size is not assessed here.

## 12. Combinations

- Stateful/warm-session direction: a CUA trajectory is a growing history behind a static prefix with tool-execution gaps; the session-affinity router built here is the primitive a KV-residency product needs, and the voice repo's arm-D idle-hint pinning (1.5x on L40S, seed 1) is the closest measured mechanism [02-evidence-ledger-voice.md vi-armD-predictive-pinning]. The stateful backend today serves a text model under the VLM's name [03-build-inventory.md].
- Document-VLM direction: same backend family, same `max_pixels` lever, same gateway; the 1024px DocVQA point (0.925, 2.7x) is already measured [evals/RESULTS.md].
- Morph-style specialized model: a CUA grounding model plus a task-specific speculator (Morph reports 3.07x with a custom draft vs 1.93x generic on codegen [05-research-morph.md]) would move differentiation from the gateway into the model; unmeasured here.
- Agent-benchmark/telemetry direction: the cost-vs-baseline headers and trajectory harness are useful even when the buyer serves the model elsewhere (BYOC mode already does this).