# H. Computer-use agent inference provider

*An OpenAI-compatible endpoint serving an open VLM at a screenshot-sized visual-token budget behind a telemetry/dedup/affinity gateway, hosted or BYOC, whose only measured advantages are the model's own tokenizer arithmetic and prefix caching vLLM already ships; the product claim ($/completed CUA task at held accuracy) is unmeasured.*

**Evidence grade D.** The claim a buyer pays for (lower $/completed CUA task at held accuracy) has never been measured on any CUA suite; no screenshot grounding number exists at the 312-322-token budget. What is measured is either a deterministic formula output (token counts from pricing.py, whose comparators are stale), a document benchmark at a different budget (DocVQA 0.925 at 1,017 tokens), a single synthetic 15-step trajectory (redundancy), a bounded/falsified encoder-reuse ceiling (~2.3x, n=8 frames, mRoPE-blocked), or vLLM prefix caching on a text model, single request, in-GPU, which the repo itself calls table stakes. The two gateway mechanisms that would be Dexa's own (exact-dup cache, session affinity) are respectively near-inert as coded and unbuilt. The "~40x" is a price model on an assumed $0.20/MTok rate. Grade D rather than C because the measured items are inputs to the thesis, not the thesis.

**Weeks to first credible public proof:** 7-9 weeks to the first credible public proof (a $/completed-task and success page on WebArena or Odysseys with 5 seeds, 5 backends, 2 screen sizes, published trajectories and per-request usage). Assumptions: one engineer full-time; the existing gateway/backend/compose reused as-is with the len/4 estimator replaced by vLLM usage fields; ~60-100 GPU-hours for the suite runs (~$150-250 at Modal A100 $2.50/h) plus ~30 GPU-h for the multi-replica affinity test and ~20 GPU-h for A100/H100 throughput; a Sonnet 5 API arm (unmeasured spend, on the order of hundreds of dollars for 100-200 tasks x 5 seeds); browser/VM infrastructure for the suite; harness bring-up taking 1-2 weeks (the most likely slip); kill gate 1 (grounding at the capped budget) reviewed at week 2-3 before any affinity work. If OSWorld-Verified is chosen instead of a browser suite, add 1-2 weeks for desktop VM infrastructure and a desktop agent harness. No design partners exist, so pilot-derived proof is not in this window.

---

## 1. Thesis

Candidate H is an OpenAI-compatible inference endpoint for screenshot-heavy computer-use agents (CUA): an open VLM served at a screenshot-sized visual-token budget, a gateway that dedups byte-identical frames, pins a session's static system/tool prefix to a warm replica, and returns per-request "what this step cost here vs. the frontier model you left" telemetry, hosted or as a bring-your-own-cloud (BYOC) recipe. The buyer runs a browser/desktop agent loop (Browser Use, Skyvern, QA/RPA agents) and pays per screenshot today. The customer's sentence: "We pointed our agent at Dexa with a base_url change, every step shows the saving, and the screenshots never left our VPC."

Honest state, verified against the repos this session: the visual-token arithmetic (322 Qwen2.5-VL vs 1,105 GPT-4o tokens per 1280x800 frame) is a formula output and belongs to the model's tokenizer, not to anything Dexa built; "~40x cheaper" is a price model on an assumed $0.20/MTok rate and stale comparators; the budget cap is inert at 1280x800; the exact-frame cache cannot fire in an append-only agent loop as coded; prefix warmth was measured on a text model, single request, in-GPU; the delta-perception prize is bounded at ~2.3x and blocked on mRoPE. No CUA-suite accuracy or $/task number exists anywhere.

## 2. Customer and workload

Who buys: teams shipping agents that act on a screen. Facts: Browser Use (112.1k GitHub stars) ships a hosted model plus open-weight `bu-30b-a3b-preview` (30B total / 3B active MoE, 2025-12-16) claimed to run "200 tasks per $1" and reports 87.4% on its Odysseys leaderboard; Skyvern (22.9k stars) accepts any OpenAI-compatible endpoint; OpenAI folded Operator into ChatGPT Agent and the Agents SDK on 2025-08-31; Cursor's Cloud Agents added computer use in February 2026 (both medium confidence) [05-research-agents.md]. Morph's Glance is in section 9. Browser Use and Skyvern are browser agents; OSWorld is a desktop suite needing its own harness.

What they run today: one screenshot per step plus a static prefix and a short action. On Anthropic's GA computer-use toolset (17 tools) a screenshot costs ~1,000-1,800 input tokens and the toolset adds ~4,500 (browser ~6,600) [05-research-agents.md]. Per-frame tokens for 1280x800 (derived from documented rules [05-research-docai.md]): Claude 28px patches 1,334; GPT-4o 85 + 170 per 512px tile = 1,105; gpt-5.x 32px patches x 1.2 = 1,200. Qwen2.5-VL bills 322 (`pricing.qwen_vision_tokens`, run this session; README says ~324). No per-image count for Qwen3-VL is recorded anywhere; a model swap changes this number.

Turn, idle and concurrency pattern: unmeasured for CUA anywhere in the repos or research files. Nearest analogs are coding agents: Mooncake's Codex traces show 5.2 s median / 81.4 s p99 tool-driven gaps; TraceLab shows a 1.4 min median human gap [05-research-agents.md]. The redundancy bench's 180 ms inter-action wait is a harness setting [evals/agent_redundancy/bench.py:64].

How they pay: per token. Sonnet 5 $2/$10 per MTok, Haiku 4.5 $1/$5, cache reads 0.1x with a 5-minute default TTL [05-research-caching.md]; gpt-5.4 $2.50/$15, gpt-5.4-nano and gpt-5.6-luna $0.20 input, gpt-4o $2.50/$10; hosted Qwen3-VL-8B ~$0.20/MTok on DeepInfra and Fireworks (secondary aggregator), Qwen3-VL-32B $0.50/$1.50 on Together [05-research-docai.md].

## 3. The pain, in the customer's words

No customer interviews exist in either repo; these are complaints inferred from documented provider behavior, written as a buyer would say them:

- "Every step re-bills a full screenshot, and on Claude that is 1,000-1,800 tokens before I've said a word."
- "My cache only works if the prefix is byte-identical, so I hand-prune screenshots every 25 turns."
- "On serverless providers the cache lives in one replica; without a session-affinity header I pay full price again." (Fireworks: caching "only works within 1 replica"; OpenAI: `prompt_cache_key` does not pin.) [05-research-caching.md]
- "I can't put customer screens through a third-party API; I need the model in my VPC." (Bland sells on-prem to voice buyers for the same reason [05-research-voice.md].)
- "I don't know what a step costs until the invoice." (No fetched provider page documents a per-request cost-vs-alternative figure; an absence, not a survey.)

## 4. Value proposition and the proof-of-value benchmark

Value proposition at the strength the ledgers support: (a) fewer billed visual tokens per 1280x800 screenshot (322 vs 1,105 GPT-4o, 1,200 gpt-5.x, 1,334 Claude: 3.4-4.1x, derived), a property of the Qwen2.5-VL tokenizer available from any Qwen host; (b) an open 7B-8B VLM's per-token price (`pricing.py` assumes $0.20/MTok; hosted Qwen3-VL-8B lists at the same $0.20); (c) session affinity so the static prefix hits vLLM's prefix cache (12x TTFT at 4k, 25-34x at 16k; text model, single request, in-GPU); (d) $0 byte-identical duplicate steps; (e) per-request savings telemetry; (f) BYOC.

What must be confronted first:

1. "~40x" is (3.4x tokens) x (12.5x assumed price) [04-conflicts.md #10]. `pricing.compare` on one 1280x800 frame, 200 text-in, 20 out gives 31.9x vs gpt-4o and 46.1x vs "claude-3-5-sonnet" (run this session), but the comparators are stale: Claude is priced at $3/$15 with a w*h/750 rule (1,366 tokens), not Sonnet 5's $2/$10 and the 28px rule (1,334). Re-based on current list prices (derived): ~30x vs Sonnet 5, ~35x vs gpt-5.4, ~3.7x vs gpt-5.6-luna or gpt-5.4-nano, ~1x vs hosted Qwen3-VL-8B. No measured serving cost per step exists.
2. The accuracy evidence is not on screenshots: Qwen2.5-VL-7B 0.925 vs GPT-4o 0.880 on 200 DocVQA pages (relaxed match) at the 1024px budget, 1,017 visual tokens per page, not 322 [ledger gpt4o-docvqa-head-to-head, BOUNDED]. Grounding accuracy at 322 tokens is unmeasured.
3. The cap is inert at 1280x800: `qwen_vision_tokens(1280, 800)` returns 322 capped or uncapped. It bites only above ~1 MP: 1920x1080 is 646 uncapped vs 312 capped; 2560x1440 is 1,196 vs 312. A 1080p screen is downscaled to ~0.69x linear scale under the cap; grounding at that scale is unmeasured. Any baseline serving the same model is token-identical to Dexa at 1280x800.
4. Redundancy prize 7.3x, realized ceiling ~2.3x: 13.7% of 28px patches change per action (2.5-6% type/edit/dropdown, ~28% select-row, 22-46% navigation) on one synthetic 15-step Playwright CRM task at 1280x800; but a 15.6% pixel change moves 52.8% of vision tokens (cos < 0.98), moved tokens' median cosine is 0.83, and the ceiling is 2.34x at cos >= 0.98, ~3.1x at >= 0.95, ~3.9x at >= 0.90 (n = 8 frames; grounding tolerance never tested) [ledgers cua-screen-redundancy, delta-perception-viability, delta-tolerance-sweep]. The 7.3x is superseded [04-conflicts.md].
5. No OSWorld/WebArena validation: `dexa_platform/README.md` 'Status' lists it as not built.
6. The "CUA-tuned backend" is `vllm serve` with the flags in the diagram (vLLM 0.24.0, Qwen2.5-VL-7B-Instruct, A100-80GB, Modal `scaledown_window=300, min_containers=0` [serve/cua_backend.py]); no custom serving code.

The proof-of-value benchmark, pre-registered:

- Metric: dollars per successfully completed task and p50 step latency at matched success, on a public suite matched to the harness: WebArena (100-200 tasks) or Odysseys with Browser Use/Skyvern; OSWorld-Verified only with OSWorld's desktop harness (founder decision). Billed tokens from vLLM `usage` fields including cached tokens.
- Setup: one harness, same tasks, five seeds, five backends, two screen sizes: 1280x800 (cap inert; isolates gateway effects) and 1920x1080 (cap bites, 646 -> 312; isolates the budget).
- Baselines: (1) vanilla vLLM 0.24.0, same model, default flags; (2) SGLang, same model; (3) hosted Qwen3-VL-8B at $0.20/MTok; (4) Sonnet 5 with the computer-use toolset and caching; (5) Browser Use `bu-30b-a3b-preview` on vLLM.
- Target: unmeasured today. Proceed: success within 3 points of native resolution at 1080p, and $/completed task at most one-third of Sonnet 5 at equal or better success. Kill: success more than 8 points below baseline (3), or cost within 1.5x of baseline (3), because then the product is a Qwen host with telemetry.
- Why a skeptical buyer believes it: harness, trajectories and per-request usage are published; cache and prefix hits are observed `usage` counts; the 1280x800 arm openly shows where Dexa equals baseline (1).

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| `gateway/app.py` (261 lines, FastAPI, v0.3.0) | custom, demo | `/v1/chat/completions`; key -> tenant; forward; exact-dup cache keyed `(img_hash, text_hash)` per session (line 209); forces `stream=False` (line 127); text tokens len/4 (line 61) |
| `gateway/pricing.py` (149 lines) | custom, 5 tests | Image-token formulas, list-price comparison; comparators gpt-4o, gpt-4o-mini, claude-3-5-sonnet (stale); Dexa rate assumed $0.20/$0.20 |
| `gateway/redundancy.py` (94 lines) | custom, 5 tests | 28px-patch luma diff (THRESH 6); `exact_duplicate` = byte-identical previous frame |
| `gateway/tenants.py`, `store.py` | custom | hosted / byoc / mock routing; in-process ledger |
| `control/*` (620 lines, 7 tests) | custom | PLG signup, hashed keys, $5 credit, 402 on exhaustion; no OAuth, no Stripe; Dockerfile omits control/ |
| `serve/cua_backend.py`, `backend_launch.sh`, `docker-compose.byoc.yml` | OSS vLLM 0.24.0 + Qwen2.5-VL-7B, custom flags | Model server; hosted scales to zero after 300 s |
| Session-affinity router; streaming; rate limits; metrics | not built | Single `GPU_BACKEND_URL` today; gaps per [03-build-inventory.md] |
| Delta-perception encoder reuse | not built; blocked | Qwen2.5-VL mRoPE ([3,1671] vs [3,881]) |

Where differentiation lives: entirely in the gateway (dedup, telemetry, tenant/BYOC routing) and operational choices (model, budget, affinity); nothing in the scheduler, connector, kernel or model is custom, and the one execution-owned lever is bounded at ~2.3x and blocked. The gateway talks HTTP to vLLM, avoiding the experimental KVConnector API, but depends on version-specific CLI flags (0.24.0 pinned; 0.28.0 current).

```
 agent harness (Browser Use / Skyvern / custom)
      | OpenAI SDK, base_url -> Dexa, x-dexa-session, x-dexa-baseline
      v
 +--------------------- gateway (custom) ----------------------+
 | key -> tenant (hosted | byoc | mock) | credit gate (opt.)   |
 | RedundancyMeter: 28px patch diff, sha256(frame)             |
 | byte-identical frame + identical text? -> cached, $0        |
 | else -> [session-affinity router: NOT BUILT] -> replica     |
 | pricing.compare(frame dims, text, out) -> x-dexa-* headers  |
 +--------------------------------------------------------------+
      v                                     v
 hosted: Modal A100-80GB (scale-to-zero)   BYOC: customer VPC, same flags
 vllm serve Qwen2.5-VL-7B --enable-prefix-caching --max-model-len 16384
   --gpu-memory-utilization 0.92 --limit-mm-per-prompt {"image":3}
   --mm-processor-kwargs {"max_pixels":1050000}
```

## 6. Evidence

### Proven

- Image-token accounting, 1280x800, as deterministic formula output with 5 passing tests: Qwen2.5-VL 322 (README ~324), GPT-4o 1,105, GPT-4o-mini 36,835 (README ~35,000) [pricing.py, run this session]; comparators stale.
- Screen redundancy: section 4 item 4, one synthetic trajectory; no OSWorld/WebArena frames [ledger cua-screen-redundancy].
- Budget vs accuracy on documents (Qwen2.5-VL-7B, vLLM 0.24.0, A100, 200 DocVQA pages, relaxed match): 1536px 0.935 (2,201 tokens, 11.4 img/s); 1024px 0.925 (1,017, 30.2 img/s, 2.7x, -1.0 pt); 768px 0.885 (574); 512px 0.735 (277); 384px 0.445 (182) [ledger doc-vlm-frontier]. Documents, not screens; 322 tokens sits between the 512px (-20 pts) and 768px (-5 pts) document points.
- vLLM prefix-cache hit vs cold TTFT (Qwen2.5-7B text, A100, single request, in-GPU): 4k ~280 -> ~24 ms (~12x); 16k ~1,250 -> ~45 ms (25-34x, 3 runs) [ledger vllm-warmstart-prefix-cache]; the repo calls it "table stakes" (1.00-1.02x vs prefix-cached SGLang) [ledger agentic-serving-value-v1].
- Build state: 29 dexa_platform tests pass; landed 2026-08-04/05; no integration test against a live backend [03-build-inventory.md].

### Bounded / contradicted

- Delta-perception reuse: BOUNDED at ~2x naive, ~2.3x with tolerance; the 5x rescue is FALSIFIED (section 4 item 4). Contradiction: `evals/RESULTS.md` 'Why incumbents can't capture it (the moat)' still claims 7.3x.
- DocVQA parity vs cost: 0.925 vs 0.880 is measured; the Qwen cost row in `modal_incumbent_docvqa.py` is hardcoded ("$0.02-0.06 / 1k pages"); GPT-4o-mini accuracy and measured $/1k pages never recorded [04-conflicts.md #10]. Contradiction: `dexa_platform/README.md` calls 40x "measured, not pitched"; FINDINGS.md calls it "copyable."
- Exact-frame cache: fires only when the frame is byte-identical to the previous frame and the concatenated message text also matches (`app.py` line 209). In an append-only loop the text hash changes every step, and even "type/edit" actions change 2.5-6% of pixels. The only exerciser, `examples/agent_loop.py`, fabricates an identical frame every third step (line 43). Near-inert on real loops as coded; hit rate unmeasured.
- Prefix warmth for CUA: the cacheable prefix is the static block (Anthropic-scale ~4.5-6.6k tokens; unmeasured for the Dexa prompt) because the 3-image limit rotates screenshots out; value per step is bounded by that prefix's cold prefill (~280 ms at 4k on A100). Multi-replica behavior and scale-to-zero cold starts are unmeasured.
- Content-aware pruning: blocked by Qwen2.5-VL mRoPE grid coupling; no accuracy numbers [ledger vlm-moat-mrope, OPEN].

### Unproven

| claim | experiment that proves or kills it | est. cost |
|---|---|---|
| Task success holds at 322 tokens (1280x800) and 312 (1080p under cap) | WebArena/Odysseys 100-200 tasks (or OSWorld with its harness), native vs capped, 5 seeds, both screen sizes | ~60-100 GPU-h (~$150-250 at Modal A100 $2.50/h) + browser/VM infra + Sonnet 5 API spend; 2-3 weeks |
| $/completed task beats hosted Qwen3-VL-8B and Sonnet 5 | Same run; bill from `usage` | included |
| Exact-dup hit rate on real trajectories is material | Replay recorded trajectories; count `served_from_cache`; also a last-frame-only key | ~2 GPU-h; 2 days |
| Prefix-cache hit rate with vs without affinity, >= 4 replicas | Consistent-hash router; `cached_tokens` and TTFT; include scale-to-zero cold starts | ~30 GPU-h; 5 days |
| Steps per GPU-hour at the CUA budget (real $/step) | Load test on A100 and H100 (voice harness needs text, tokenizer and image payloads first) | ~20 GPU-h; 1 week |
| Model swap keeps the token advantage | Per-frame tokens for Qwen3-VL-8B/30B-A3B and bu-30b from `usage` | ~1 GPU-h; 1 day |
| Delta reuse at cos >= 0.98 preserves grounding | ScreenSpot-style click grounding, reused vs fresh embeddings; needs an mRoPE-aware forward pass | ~60 GPU-h + 2-3 engineer-weeks |

## 7. MVP and 6-week build plan

Ships first: hosted endpoint plus BYOC compose, observed telemetry, streaming, session affinity, public benchmark page. Reused: the section 5 components, `evals/agent_redundancy/*`, `evals/modal_vlm_frontier.py`, `evals/modal_incumbent_docvqa.py`; from voice-inference, the Modal sweep pattern (`evals/modal_arm_a.py`) and metrics pipeline (`vkv/metrics/`), which send synthetic token ids and must be generalized [03-build-inventory.md].

- Week 1: Harness and corpus. Pick suite (section 11); wire Browser Use and Skyvern to the gateway; stand up WebArena (or OSWorld VMs); record trajectories. Refresh `pricing.py` comparators. Slip risk: suite setup alone is commonly a week.
- Week 2: Kill gates 1 and 3: task success native vs capped at both screen sizes (5 seeds); exact-dup hit rate on recorded trajectories. Replace `app.py`'s len/4 estimate and modeled cost with vLLM `usage` fields.
- Week 3: Baselines (2)-(5). Kill-gate review before building affinity: if gate 1 fails, stop.
- Week 4: Session-affinity router (consistent hash on `x-dexa-session`, least-loaded fallback) across >= 4 replicas; measure `cached_tokens` with and without affinity, including cold starts. SSE streaming.
- Week 5: Throughput and $/step on A100 and H100; harden BYOC compose (include control/); add `/metrics`.
- Week 6: Public benchmark page with pre-registered numbers, trajectories and harness. Design-partner outreach begins here, not pilots: no customer contact exists yet.

Realistic first credible public proof: weeks 7-9, because the harness and multi-seed runs are the items most likely to slip. Out of scope: Stripe, OAuth, Cloudflare edge, delta-perception engine.

## 8. Pricing model

Options the architecture makes expressible:

1. Per step (per screenshot) flat price: visual tokens per frame are deterministic given the budget (322 for 1280x800; 312 for 1080p under the cap), so a step is a bounded unit.
2. Per token at a blended open-VLM rate (`pricing.py` assumes $0.20/$0.20; aggregators list hosted Qwen3-VL-8B at $0.20). Least differentiated.
3. Cached steps at $0 and prefix-hit tokens discounted (precedents: Tensormesh $0 cached input; Fireworks 50%; Anthropic 0.1x; Morph and Fireworks sticky-session headers [05-research-kvcache.md; 05-research-caching.md; 05-research-morph.md]). Expressible only after the affinity router exists and the exact-dup cache is re-keyed to fire.
4. BYOC: recipe-plus-telemetry license; the customer pays their GPU (Modal A100 $2.50/h, H100 $3.95/h; section 11 has other providers).

Cost side: unmeasured for screenshots. The only throughput point is 30.2 document pages/s at 1,017 tokens on one A100, turned into a hardcoded "$0.02-0.06 / 1k pages" (cost model assumes $1.80/GPU-h) [evals/RESULTS.md; evals/stateful_cost_model.py:20]. No price can be set before the Week 5 steps/GPU-hour number.

## 9. Competitive facts

| who | adjacent thing they ship | what the research files do not show them shipping | source |
|---|---|---|---|
| Anthropic | GA computer-use toolset (17 tools), screenshots ~1,000-1,800 tokens, 5-min/1-h cache TTL; Sonnet 5 $2/$10 | Per-screenshot flat price; self-hosted/BYOC; per-request cost-vs-alternative telemetry | 05-research-agents.md; 05-research-caching.md |
| OpenAI | CUA in ChatGPT Agent and Agents SDK; patch image tokens (1024x1024 = 1,229 on gpt-5.4); gpt-5.6-luna and gpt-5.4-nano $0.20/MTok input; `prompt_cache_key` routing | Machine-pinned caching; BYOC | 05-research-agents.md; 05-research-docai.md; 05-research-caching.md |
| Google Gemini 3 | `media_resolution` low/medium/high/ultra = 280/560/1,120/2,240 tokens per image; Gemini 3.8 Flash $0.75/MTok input | CUA-specific endpoint or per-step billing | 05-research-docai.md |
| DeepSeek | `deepseek-v4-flash-vision-exp` (2026-08-21): images up to 384 tokens at V4-Flash prices ($0.44/MTok miss peak, $0.22 off-peak, $0.014 hit) | Self-hosting; session affinity guarantees | 05-research-caching.md |
| Browser Use | Hosted ChatBrowserUse model; open `bu-30b-a3b-preview`; "200 tasks per $1"; 87.4% Odysseys | An inference API for third-party agents; BYOC recipe | 05-research-agents.md |
| Morph | Glance (`morph-computer-use-v0`, "10x cheaper than general-purpose"); sticky KV via `x-session-id`; prefix caching on all models | Fetched pricing or benchmark for Glance; BYOC | 05-research-morph.md |
| Fireworks / DeepInfra / Together | Hosted Qwen3-VL (8B ~$0.20/MTok per aggregators; 32B $0.50/$1.50); Fireworks `x-session-affinity`, 50% cached discount, BYOC private preview | Screenshot-specific budgets; per-step telemetry | 05-research-docai.md; 05-research-providers.md |
| Tensormesh | Cached input billed at $0; Qwen3 30B $0.15/$0.60 | VLM/CUA focus; on-prem "Operator" is "Coming Soon" | 05-research-kvcache.md |
| vLLM / SGLang | Prefix caching; video-token pruning (EVS); open RFC #45098 (2026-06-10) for image pruning; SGLang EVS incompatible with Qwen2.5-VL | Shipped image-token pruning; in-engine per-session affinity | 05-research-docai.md; 05-research-kvcache.md |

Derived arithmetic (list prices x documented token rules, not measurements): input cost per 1280x800 frame is ~$0.000064 at Dexa's assumed $0.20/MTok, ~$0.00024 on gpt-5.6-luna (1,200 x $0.20), ~$0.00021 on Gemini 3 low (280 x $0.75), ~$0.00017 peak / ~$0.000084 off-peak on DeepSeek (384 tokens). The modeled gap versus frontier small-image modes is 1.3-3.7x, not ~40x; their CUA accuracy is unmeasured.

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| Screenshot budget breaks grounding (1080p under cap = 0.69x scale) | Suite success, native vs cap, both screen sizes, 5 seeds | > 8 pts drop | <= 3 pts drop |
| Benchmark isolates nothing at 1280x800 (baseline (1) token-identical) | Include the 1080p arm; label the 1280x800 arm gateway-only | 1280x800-only benchmark | both arms published |
| Product is a Qwen host with telemetry | $/completed task vs hosted Qwen3-VL-8B at $0.20/MTok | within 1.5x | <= 0.7x hosted Qwen and <= 1/3 Sonnet 5 at equal success |
| Exact-dup cache never fires | `served_from_cache` on recorded trajectories, current and last-frame-only keys | < 3% of steps for both keys | >= 10% of steps |
| Affinity does not move prefix hits | `cached_tokens` share with vs without router, >= 4 replicas, plus cold-start rate under scale-to-zero | < 10 pts gain or > 5% cold-start steps | >= 30 pts gain |
| $/step economics | steps per GPU-hour at p95 step latency <= 2 s on A100/H100 | implied cost > $0.20/MTok-equivalent | <= 0.5x hosted Qwen price at 50% utilization |
| Model swap erases the token advantage | per-frame tokens from `usage` for Qwen3-VL / bu-30b | >= 1,000 per 1280x800 frame | <= 400 |
| Delta-perception R&D | grounding accuracy with cos >= 0.98 reuse | > 2 pts drop or reuse < 1.5x | <= 1 pt drop at >= 2x (ceiling 2.34x) |
| vLLM flag drift (0.24.0 pinned, 0.28.0 current) | BYOC compose on current vLLM | multimodal flags removed without replacement | flags or equivalents present |

Thresholds are pre-registered choices, not measured values. Frontier small-image modes (Gemini 280, DeepSeek 384, luna/nano $0.20) are a founder decision, not a measurement.

## 11. Founder decisions

- Model: Qwen2.5-VL-7B (deployed; 0.925 DocVQA at 1024px) vs Qwen3-VL-8B (DocVQA 96.1 per its report, Apache-2.0) vs Qwen3-VL-30B-A3B vs Browser Use's CUA-trained 30B-A3B. Evidence: Week 2 task success and per-frame tokens per model; no CUA-suite number exists for any.
- Suite and harness: WebArena/Odysseys with browser harnesses vs OSWorld-Verified with a desktop harness; success definition; seed count that counts as proof.
- Screen-size policy: cap 1080p+ screens to ~312 tokens (cheaper, grounding unmeasured) or serve native (646-1,196 tokens, no budget advantage).
- Hosted vs BYOC vs OSS gateway; hosted scale-to-zero (current) vs always-warm replicas for prefix affinity. BYOC removes the data-residency objection but leaves telemetry as the only paid artifact.
- GPU class: A100 ($1.39-2.79/h across listed neoclouds; AWS p4de $3.43), H100 ($1.73-12.29/h), L40S ($0.79-2.25/h), MI300X ($2.59-6.00/h) [05-research-gpu-pricing.md]; decided by Week 5 steps/GPU-hour.
- Whether frontier adjacency matters: Gemini 3's 280-token mode, DeepSeek's 384-token images, luna/nano at $0.20/MTok, Anthropic's toolset and caching, Morph's Glance. Facts and derived per-frame costs are above.
- Whether to fund the delta-perception engine given the ~2.3x ceiling and the mRoPE wall, or treat ~2x as sufficient.
- Whether to re-key the exact-dup cache to the last frame only, accepting that a repeated frame with new history may deserve a new answer.
- Pricing unit: per step vs per token vs $0-cached; publish the baseline price table or let buyers set it.
- Whether the "cheaper-VLM reseller" framing FINDINGS.md rejected is acceptable as a wedge. Market size is not assessed here.

## 12. Combinations

- Stateful/warm-session direction: the affinity router built here is the primitive a KV-residency product needs, and the voice repo's arm-D idle-hint pinning (1.5x on L40S, seed 1, 8 ms full-load-window margin) is the closest measured mechanism [02-evidence-ledger-voice.md vi-armD-predictive-pinning]. The stateful backend today serves a text model under the VLM's name [03-build-inventory.md]. A session API that stops resending history would also fix the exact-dup cache's text-hash problem.
- Document-VLM direction: same backend, same `max_pixels` lever, same gateway; the 1024px DocVQA point (0.925, 2.7x) is already measured.
- Morph-style specialized model: a CUA grounding model plus a task-specific speculator (Morph reports 3.07x with a custom draft vs 1.93x generic on codegen [05-research-morph.md]) would move differentiation into the model; unmeasured here.
- Agent-benchmark/telemetry direction: the cost-vs-baseline headers and trajectory harness stand alone (BYOC mode already does this).
