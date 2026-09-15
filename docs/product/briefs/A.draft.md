## 1. Thesis

A hosted inference provider for open-weight models whose unit of service is a durable, warm agent session, not a stateless request. The customer opens a session; the provider keeps its KV cache resident across idle gaps of minutes to hours (HBM while active, pinned host RAM, then NVMe as the gap grows, dropped past a measured break-even) and bills each turn as residency time plus the tokens actually added, instead of re-billing a 100K-token prefix. The restore-vs-prefill physics in the repos is real at single-request scale and in-engine to 16k. The product hypothesis, that an idle-aware tiering policy beats stock LRU offload at serving concurrency and agent-scale context, is designed but unmeasured (04-conflicts.md #3, #6). This brief exists to get that hypothesis measured in six weeks.

## 2. Customer and workload

Customer: teams running coding, computer-use, or human-in-the-loop agents on open-weight models, either self-hosting on vLLM/SGLang (OpenHands recommends Qwen3.6-35B-A3B; Browser Use ships a 30B open model; Cursor's Composer 2 started from Kimi K2.5; open models carry a majority of OpenRouter tokens, concentrated in coding and agentic work: 05-research-agents.md) or buying from generalized providers whose cache does not survive their gaps.

Workload shape, all external (05-research-agents.md):

- Per step: median prefix 126,180 tokens (Claude Code) / 115,584 (Codex); median new input 857/886; median output 252/184 (TraceLab, arXiv 2606.30560). In 219 production Claude Code sessions, median request 195K input / 317 output; 96% of requests reuse >=90% of input as verbatim prefix (llm-d GLM-5.2 post).
- Gaps: human thinking gap median 1.4 min, p90 20.6 min (TraceLab); tool-driven gaps median 5.2 s / P99 81.4 s (vLLM x Mooncake); inter-turn pauses median 2 s / p99 11.4 min (llm-d); the 99.9th-percentile Claude Code turn grew from under 25 min to over 45 min (Anthropic).
- Cache behaviour: global prefix hit 95.7% but only 84.4% on user-initiated steps (TraceLab). Misses concentrate on exactly the human-gap turns.
- Computer-use adds ~1,000-1,800 input tokens per screenshot and ~4,500 for the toolset (Anthropic docs).

Per-customer gap distributions are unmeasured beyond these samples; collecting them is week-1 work.

## 3. The pain, in the customer's words

- "After a gap longer than the TTL, the next request recomputes the full input and re-establishes the cache"; API-key users default to 5 minutes (https://code.claude.com/docs/en/prompt-caching).
- 17.1% overpayment ($949 on a Sonnet 4.6 sample, $1,581 on Opus 4.6) reconstructed from 119,866 Claude Code calls after writes moved from 1h to 5m in March 2026; closed "not planned" (https://github.com/anthropics/claude-code/issues/46829).
- Codex on Azure past ~150K tokens: cached_input_tokens = 0; one run ~$15.9 for ~20M tokens with 3.28M re-billed tokens = 71% of fresh input; no passthrough for prompt_cache_retention (https://github.com/openai/codex/issues/25604).
- "prompt_cache_key ... do not pin requests to a machine or guarantee a cache read hit" (OpenAI); "Cache entries can be evicted at any time due to server load or restarts" (xAI); Fireworks caching "only works within 1 replica" (05-research-caching.md, 05-research-providers.md).
- "At step 10 of a typical agent loop the model processes ~11,500 input tokens to act on ~200 new tokens" (Tensormesh blog).
- The maintainers state the gap themselves: orchestrators "know session structure and tool-gap duration that request-local LRU cannot observe" (SGLang RFC #27574).

## 4. Value proposition and the proof-of-value benchmark

Proposition: your agent's session stays warm through the gaps your provider's cache does not survive; resume is lossless; you pay residency plus delta tokens.

The proof benchmark is the crossover-surface experiment that 04-conflicts.md #1, #4, #5, #6 and #11 each call for, run once so it settles them.

**IGR-1 (Idle-Gap Resume).** Harness: vLLM's own `benchmarks/multi_turn/benchmark_serving_multi_turn.py --max-active-conversations` (docs/BENCHMARK_PLAN.md), or the voice-inference harness generalized to text (03-build-inventory.md lists the gaps). Sessions replay gaps sampled from TraceLab (median 1.4 min, p90 20.6 min) and llm-d (p99 11.4 min), with ~860 new-input / ~200-250 output tokens per turn.

Model: Llama-3.1-8B-Instruct (the model of most repo evidence) plus one model a design partner actually runs. GPUs: A100-80GB (continuity), H100-80GB and one H200 point (conflict #5). Grid: context {16k, 32k, 64k, 128k} x active sessions {1, 8, 32, 128} x 3 seeds. Routing defeats the local prefix cache or caching is OFF (BENCHMARK_PLAN gotcha #1), plus one run with caching ON and live KV larger than the HBM pool (voice arm A collapsed when ~42 GB live KV exceeded a ~26 GB pool).

Arms, versions pinned: **A** vanilla vLLM prefix caching; **B** vLLM + current LMCache CPU LRU, stock; **C** LMCache CPU + local-disk LRU; **D** ours: contiguous pinned-RAM connector with async load plus enforced idle-aware demotion RAM -> NVMe -> drop driven by client idle hints; **E** = D with fp8 KV.

Metrics: p50/p99 TTFT on resumes after gaps over 5 min, on the full-load window, not whole-run (voice recomputation: whole-run p95 understated full-load p95 by 20-25 ms); aggregate restore GB/s; GPU-seconds per turn; GB-hours held per tier; greedy-output identity vs cold prefill on every resume.

Primary metric and target: p99 resume TTFT at 64k, 32 active sessions, H100, arm D vs arm A: **>= 5x**. The in-engine single-request number is 17.1x at 16k (lmcache-offload-restore); the 16k-64k RAM tier under load has never been measured (conflict #1), so 5x is a deliberately conservative pre-registration. Secondary: arm D holds >= 30% fewer GB-hours than arm B at equal p99 (the policy delta, conflicts #3/#6). Tertiary: 100% greedy identity.

Why a skeptical buyer would believe it: independent harness; every version pinned (the repos never ran the same LMCache version: voice 0.3.2 on vLLM 0.9.2, dexa unpinned on 0.24.0); raw events and manifests committed with connector provenance (every voice manifest records git_sha "unknown" and no connector field); two GPU generations; three seeds; a correctness column; $ from the invoice, not the $1.80/hr constant in evals/stateful_cost_model.py.

## 5. Architecture

| name | custom or OSS | role |
|---|---|---|
| Session API (create / turn / resume / close, streaming) | custom, seed from dexa_platform/sessions | Session is the addressable primitive; carries idle hint and tier policy per turn |
| Session-to-replica affinity + admission | custom (new) | Routes a session to the replica holding its KV; admits by resident-set budget (today a single fixed URL) |
| vLLM serving tier | OSS vLLM | Prefill/decode, paged KV, prefix cache for the active window |
| Session KV connector | custom, seed from voice `vkv/backends/voice_kv_connector.py` | Contiguous pinned-slab save/restore, async layer-pipelined load, block-hash index, delay-free pinning from idle hints |
| Tiering policy engine (enforcing) | custom, seed from dexa_platform/sessions/tiering.py | Demotes HBM -> RAM -> NVMe -> drop from measured break-evens and idle predictions; today advisory only |
| NVMe / object tier | OSS (vLLM OffloadingConnector filesystem tier or LMCache disk backend) | Cold residency for hour-scale gaps |
| Portable KV blob + correctness probe | custom, src/dexa/session (DEXAKV01) + voice `evals/modal_kv_correctness.py` | Migration, failover, bit-identity checks |
| Metering / keys / credits | custom, dexa_platform/control | Residency GB-hours per tier + delta tokens |
| Benchmark harness | OSS vLLM multi-turn bench + voice `vkv/` | IGR-1 and the public leaderboard |

Differentiation lives in the tiering policy and the session-semantic hint protocol (the one mechanism that moved a knee in either repo, voice arm D), and in the residency pricing that policy enables. Not in restore: LMCache, vLLM OffloadingConnector, Mooncake and Dynamo KVBM all restore. Built on vLLM unchanged through the stock KVConnectorBase_V1 seam, no fork. Replaced: the eviction policy (LRU/ARC/LFU in every OSS store, 05-research-kvcache.md) with an idle-window-aware one, and the request-scoped API with a session-scoped one.

```
 client agent ──(session_id, delta tokens, idle_hint)──▶ Session API / affinity router
                                                              │ metering: residency GB-hr + delta tok
                                                              ▼
                                   ┌──────── GPU replica (vLLM + session connector) ────────┐
                                   │  HBM: active sessions (prefix cache, pinned by hint)   │
                                   │   │ demote (idle > t_hbm)        ▲ restore ~10-20 GB/s │
                                   │   ▼                              │ contiguous, async   │
                                   │  pinned host RAM slab (content-addressed sessions)     │
                                   └──────┬────────────────────────────▲──────────────────┘
                                          │ demote (idle > t_ram)      │ restore ~3 GB/s naive; direct-IO unmeasured
                                          ▼                            │
                                       NVMe / object tier ──(idle > t_nvme)──▶ drop, re-prefill on return
```

## 6. Evidence

### Proven

- Restore beats re-prefill single-request, in-engine, RAM tier: vLLM 0.24 + LMCache CPU, Qwen2.5-7B, A100: 4k 301 -> 30 ms (10.0x); 16k 1,320 -> 77 ms (17.1x); 16,384 tokens in 39.8 ms at 22.0 GB/s (01 lmcache-offload-restore).
- HF-level physics to 64k: prefill 296 / 1,321 / 3,167 / 8,567 ms vs pinned-CPU restore 25 / 115 / 182 / 297 ms at 4k/16k/32k/64k (11.8x -> 28.9x); NVMe via naive torch.load 246-1,228 ms (01 stateful-warm-session-hf). HF prefill is ~5-7x slower than vLLM at 8k (docs/RESULTS.md: 617 ms vs 3.3-4.2 s), so ratios against the engine are smaller.
- Lossless, portable resume: bit-identical output across two vLLM processes at TP=1; identical_output=true at 8k-64k (01 cross-instance-resume; 04 fact (b)).
- fp8/int8 KV at 0.5x bytes with 100% greedy agreement over 48 tokens, one passage (01 kv-interchange-formats).
- Contiguous pinned-RAM restores sustain concurrency where chunked blocking loads collapse: VoiceKVConnector N=300 p95 316.8 ms stable vs LMCache 0.3.2 default 25,997.7 ms / layerwise 6,025.1 ms; L40S, ~2.5k-token sessions, seed 1 (02 vi-contiguous-vs-chunked-transfer).
- Session-semantic residency hints move a knee: arm D N=300 p95 272.2 ms PASS vs B 316.8; 1.5x, single seed, 8.1 ms margin on the full-load window (02 vi-armD-predictive-pinning).
- End-to-end demo: 12k session 5.4 s cold -> 1.5 s warm (3.7x), one run (01 stateful-session-service-live).

### Bounded / contradicted

- Crossover is a property of tier bandwidth, not of KV loading (conflict #1): pinned RAM at ~9-22 GB/s crosses over below 4k for 7-8B on A100; Dexa's disk-backed connector at ~0.6-0.9 GB/s lost 0.37x at 8B/8k (1,681 vs 617 ms) and crossed near 64k; under 16-24-way concurrency it lost 1.6-2.2x at p99 even with async load (01 connector-under-concurrency-sync-async). The 16k-64k RAM tier under concurrency was never measured. The README's HF-eager-baselined speedup is superseded and not cited.
- "Three independent reproductions" (conflict #3): two are in-GPU prefix caching and stock LMCache; the claimed delta, the tiering policy, is advisory only (edge/README.md; sessions/tiering.py has no enforcement path).
- "Latency win unconditional" (conflict #4): single-request; the deployed substrate's concurrent behaviour at 16k-64k is unmeasured; the only concurrent RAM-tier data is 2.5k-token voice sessions at ~1.5 GB/s aggregate; the arm D margin sits inside unmeasured seed variance.
- Cost model (conflict #5): "2-6x cheaper" and break-evens (64k: HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr) are modeled at $1.80/hr A100, $0.006/GB-hr RAM, $0.0002/GB-hr NVMe from HF single-request timings. Warm break-even = prefill_s x usable_HBM_GB / (kv_GB x 3600): a faster GPU shrinks it, larger HBM grows it. No H100/H200/B200 measurement exists; both repos name this the top threat.
- In-engine evidence stops at 16k (conflict #11): >=32k crashed vLLM V1 EngineCore in that harness (YaRN, three configs). All 32k-128k numbers are HF-only; the median agent prefix is ~116-126K.
- 08-02 "every path inside the inference stack is closed for a small team" vs 08-05 "the thesis is a business" (conflict #6) is unreconciled; IGR-1 arm B vs D is the resolving test.

### Unproven

| claim | experiment | est. cost |
|---|---|---|
| RAM-tier restore beats re-prefill at 16k-64k under 8-128 concurrency | IGR-1 arms A/B/D, A100 | ~240 GPU-hr ~ $600 (Modal A100 $2.50/hr) |
| Idle-aware tiering holds fewer GB-hours than LMCache LRU at equal p99 | IGR-1 arm B vs D with real gap replay | included above |
| The win survives H100/H200 (faster prefill, PCIe Gen5, more HBM) | IGR-1 H100 + one H200 point | ~240 GPU-hr ~ $950 (Modal H100 $3.95/hr) + ~$50 |
| Resume works in-engine at 64k-128k | native-128k model, chunked prefill on | ~2 GPU-days |
| NVMe direct-IO restore GB/s and break-even; 2-replica migration under traffic | O_DIRECT/GDS path vs torch.load; round-robin IGR-1 (gotcha #1) | ~3 GPU-days |
| Buyers pay residency + delta; $ saving is material at open-model list prices | 3 design partners on the MVP for 2 weeks; measured $/session-hour | staff time + IGR-1 |

## 7. MVP and 6-week build plan

MVP: a session API over vLLM on rented A100/H100 with an enforcing RAM+NVMe tier, an idle hint on every turn, observed warm/cold reporting, residency metering, and published IGR-1 results with raw logs.

- **Week 1.** Three design-partner conversations; capture per-session gap histograms. Generalize the voice harness (`vkv/loadgen/runner.py`, `vkv/metrics/*`, `vkv/llm/vllm_engine.py`, `evals/modal_arm_a.py` sweep driver) to text and the chat endpoint; add manifest provenance. Pin vLLM, LMCache, torch.
- **Week 2.** Connector v0: port `vkv/backends/voice_kv_connector.py` (574 LOC; TP=1, fp16, prompt-KV only, sync loads, vLLM 0.9.2) to current KVConnectorBase_V1 ("experimental and subject to change", 05-research-kvcache.md). Add async layer-pipelined load, decode-KV save, block-hash keying (dexa's whole-prompt keying scored zero hits on prefix_repetition), tenant-scoped keys. Run `evals/modal_kv_correctness.py` and commit its output (never archived).
- **Week 3.** IGR-1 on A100 at 16k/32k/64k, arms A/B/D. Clear the in-engine >=32k crash with a native-128k model and chunked prefill (conflict #11). First kill-gate read.
- **Week 4.** Enforcing tiering: NVMe secondary tier; demotion driven by the arm D hint protocol (~40 lines connector + 6-line client hint, 03-build-inventory.md) and measured break-evens; GB-hours metering. Arms C and E.
- **Week 5.** H100 grid and one H200 point; 128k; direct-IO NVMe. Publish IGR-1 with raw events.
- **Week 6.** Product surface: reuse `dexa_platform/sessions/{service,store,tiering}.py` (383 LOC) and `dexa_platform/control/*` (620 LOC keys/metering/credits); fix known gaps: streaming (gateway forces stream=False; SessionDO hard-codes max_tokens 16), warm/cold observed from connector counters (today warm = not first_touch), affinity map, durable session store, tenant-isolated KV. Design partners on the endpoint.

Reused: voice harness and connector; `serve/vllm_lmcache_backend.py` (the config behind 10.0x/17.1x); `evals/modal_lmcache_restore.py`, `modal_stateful_session.py`, `stateful_cost_model.py` (rewired to invoice rates); `src/dexa/session/{blob,store}.py`; `scripts/modal_bench_contention.py`. Not reused: `src/dexa/engine/vllm_connector.py` disk path (0.37x), compaction, cartridges, edge/ until a live backend exists. New: affinity router and admission, enforcing policy engine, NVMe tier, streaming session API, GB-hour metering, provenance-complete benchmark publication.

## 8. Pricing model

Residency + delta tokens, four meters:

1. Delta input tokens (new this turn) at a list per-token rate.
2. Resident prefix tokens on resume: $0 (Tensormesh bills cached input at $0) or a low read rate.
3. Residency per tier: HBM free while active up to a short window; RAM and NVMe metered per GB-hour (or per 1M-token-hour, Gemini's shape for explicit caches at $0.50-$4.50 per 1M tokens per hour); sessions drop past a customer-set TTL.
4. Output tokens at list.

Cost floor, derived from cited rates: Llama-8B bf16 KV is 131 KB/token (docs/RESULTS.md persist bench), so 64k = 8.4 GB and 128k = 16.8 GB; host RAM at $0.0036-0.008/GiB-hr (AWS proxy / Modal, 05-research-gpu-pricing.md) puts a 128k session at ~$0.06-0.13 per RAM-hour; NVMe at ~$0.0001/GB-hr puts it at ~$0.002/hr; fp8 halves both. Qwen2.5-7B is 57 KB/token (3.76 GB at 64k). Comparators: Anthropic 1h cache write = 2x base input (a 128k write on Sonnet 5 at $2/M is $0.51 vs $0.26); DeepInfra 1h retention 2x write, 0.2x read; Managed Agents $0.08/session-hour while running.

Caveat: at open-model list prices the avoided re-prefill is small in dollars for small models (64k on DeepInfra Llama-8B at $0.02/M is ~$0.0013; on 70B at $0.10/M ~$0.0064), so the buyable unit for 7-8B may be latency and capacity rather than dollars; the dollar case grows with model size and is unmeasured. Willingness to pay per GB-hour is unmeasured.

## 9. Competitive facts

| who | what they ship that is adjacent | what they do not ship per the research files | source |
|---|---|---|---|
| Anthropic | 5-min cache refreshed on use; 1h TTL at 2x write; Managed Agents stateful sessions at $0.08/session-hour while running | Retention beyond 1h; documentation of whether the model-side cache stays warm during an idle Managed Agents session | platform.claude.com pricing, prompt-caching, managed-agents |
| OpenAI | 30-min TTL (GPT-5.6+); 24h retention pre-5.6 with KV offloaded to GPU-local storage; previous_response_id | Machine pinning or guaranteed hits; a retention price | developers.openai.com prompt-caching, your-data |
| Google Gemini | Explicit caches billed per 1M-token-hour ($0.50-$4.50), 1h default TTL; 55-day Interactions state | Explicit caching inside the Interactions API | ai.google.dev caching, pricing, interactions |
| DeepSeek | Disk cache, free storage, cleared "within a few hours to a few days", hits ~3% of miss price | Guaranteed hits; documented eviction policy | api-docs.deepseek.com |
| Fireworks | Per-replica cache, x-session-affinity, 50% cached discount, "several minutes... up to several hours" | Cross-replica KV movement; a retention SLA | docs.fireworks.ai prompt-caching |
| DeepInfra | Explicit 5m/1h retention at 1.25x/2x write, 0.2x read, 8,192-token increments, on two models | Retention beyond 1h; tiered pricing | docs.deepinfra.com prompt-cache-retention |
| Baseten | KV-aware routing over a real-time index of worker cache contents; cached input at 10% | Documented TTL, session affinity, retention pricing | docs.baseten.co bis-llm |
| Morph | Automatic prefix caching "with per-request TTL control", sticky placement via x-session-id, 50% standby tier | Published TTL bounds; tier pricing | docs.morphllm.com fast-models |
| Tensormesh | LMCache-based SaaS, cached input $0, Operator "coming soon", post-v1 price = 30% of estimated savings; $24.5M raised | Idle-window or TTL eviction (LMCache documents LRU/IsolatedLRU/noop only); named customers | tensormesh.ai pricing, faq; docs.lmcache.ai |
| LMCache / Mooncake / Dynamo KVBM / vLLM OffloadingConnector / llm-d / SGLang HiCache | Multi-tier offload (CPU, NVMe, S3, P2P), LRU/ARC/LFU eviction, Mooncake soft-pin 30 min default / 24h max, llm-d 48K pull 235 ms vs 1,988 ms recompute | A hosted multi-tenant session product; idle-window-aware eviction (SGLang RFC #27574, Mooncake PR #2835 are open/draft) | github.com LMCache, Mooncake, ai-dynamo, llm-d, sglang issues 27574/24656 |
| Continuum; "Stateful Inference" (research) | TTL-pinned KV from reload cost and queueing delay, >8x JCT; persistent KV across turns, 2.1-4.2x per turn | A product or hosted service | arXiv 2511.02230, 2605.26289 |

## 10. Risks and pre-registered kill gates

| risk | measurement | number that kills | number that proceeds |
|---|---|---|---|
| RAM-tier win vanishes under concurrency at agent context | IGR-1 p99 resume TTFT, 64k, 32 sessions, A100, arm D vs A, 3 seeds, full-load window | < 2x | >= 5x |
| Policy adds nothing over stock LMCache LRU (conflict #6) | GB-hours held, arm D vs B, at equal p99 and hit rate | < 15% reduction | >= 30% reduction |
| Bigger/faster GPU moves the wall (conflict #5) | H100 ratio (arm D vs A) relative to A100 ratio; H200 point | H100 ratio < 2x absolute | >= 60% of A100 ratio and >= 3x absolute |
| In-engine long context does not run (conflict #11) | vLLM + connector at 128k, chunked prefill | not running by end of week 3 | 128k resume measured with identity |
| Restore is lossy | greedy identity on every resume, 64k-128k | any divergence removes "lossless" from all copy | 100% identity |
| Pinned-slab advantage was LMCache-version-specific | arm D vs B at the same pinned current LMCache | D p99 within 10% of B | D p99 <= 0.5x B where B degrades |
| NVMe tier never pays | direct-IO restore GB/s; break-even at invoice rates | NVMe restore slower than re-prefill at 64k on H100 | >= 3 GB/s and break-even >= 1 hr |
| Customers' gaps are shorter than provider TTLs | design-partner gap histograms | p90 gap < 5 min for all 3 partners | at least one partner with p50 gap > 5 min |
| Nobody buys residency | 3 design partners on MVP, 2 weeks | 0 of 3 accept metered residency | >= 2 of 3 run production traffic |

## 11. Founder decisions

1. **Model class.** All repo evidence is 7-8B dense. Customers' agents run 30B-A3B to 2.8T MoE (Kimi K3, GLM-5.2/5.3; llm-d notes MLA KV is 576 FP8 elements/token and TP=8 keeps eight KV copies). Options: prove on 8B then scale; start on a partner's model. Evidence: KV bytes/token and prefill tok/s for the target model; IGR-1 on it.
2. **Hosted provider vs operator/plugin.** README (Apache-2.0 sidecar, hosted SaaS out of scope) contradicts FINDINGS/platform/edge (hosted PLG) (conflict #7). Tensormesh's Operator is "coming soon"; SGLang and Mooncake are adding TTL-bounded pins in the open. Options: hosted only; operator for self-hosters; both. Evidence: whether design partners self-host (no public survey exists).
3. **Substrate.** LMCache (current product; 10-17x single-request; collapsed at N=300 on 0.3.2), custom connector (voice; held N=300), vLLM native OffloadingConnector (LRU/ARC; FS/S3/P2P tiers; TieringManager RFC open), SGLang HiCache (L2 private per instance). Evidence: IGR-1 arm B vs D on pinned current versions.
4. **Frontier-model customers.** The loudest pain (section 3) is on Claude/Codex, which this provider cannot serve. Options: teams already on open models; teams converting because of TTL economics. Evidence: partner conversations; open-weight share data (Mozilla, OpenRouter).
5. **Pricing shape.** Per-GB-hour, per-1M-token-hour (Gemini), $0 cached with per-token markup (Tensormesh), flat subscription (Wafer Pass $10-25/week). Evidence: partner reaction; measured $/session-hour from IGR-1.
6. **Compute sourcing.** Rent per-second (Modal A100 $2.50, H100 $3.95, H200 $4.54/hr; RunPod H100 $2.69-3.29) vs commit (SF Compute $1.90 average). Host RAM per GPU spans 125-283 GB (RunPod) to 2,048 GB per 8-GPU CoreWeave node and bounds resident sessions.
7. **Stay in the stack or go up-stack.** The repo's 08-02 verdict says up-stack; its 08-05 verdict says this is a business. Evidence: IGR-1's secondary metric (policy delta) is the deciding number.

## 12. Combinations

- **With the voice candidate.** The arm D hint protocol is the same mechanism at 2.5k tokens and 25 s idle; here it is 100k+ tokens and 1-20 min idle. One connector, one policy engine, two hint calibrations.
- **With the portable KV blob (README direction).** Cross-instance bit-identical resume (proven, TP=1) becomes migration and failover between replicas and clouds, and a customer-exportable session artifact.
- **With fp8 KV (kv-interchange-formats).** Halves residency bytes and doubles every modeled break-even; arm E measures it.
- **With a Morph-style specialized model.** A compaction or fast-apply subagent served inside the same warm session avoids re-shipping the prefix to a second provider; Morph's Compact (33,000 tok/s, line-deletion, byte-identical survivors) shows the shape.
- **With SGLang RFC #27574 / Mooncake PR #2835.** Upstream the hint envelope (retention: prefix_tokens, ttl_seconds) so the client protocol is standard and the hosted policy is the product.
