# C. Session-aware serving engine: one residency scheduler for every idle regime (hosted + BYOC)

*A hint-driven KV residency scheduler (pin / RAM / NVMe / drop per session) inside stock vLLM, sold hosted and BYOC, whose one in-engine win (voice arm D, 300 vs 200 sessions per L40S) is single-seed, synthetic-trace and 8 ms from failing, and whose agent-regime claim has no in-engine measurement above 16k tokens.*

**Evidence grade C.** The differentiating claim (a return-time hint beats reactive/LRU residency at matched memory) is measured exactly once in-engine and the ledger labels it BOUNDED: voice arm D, seed 1, L40S 46 GB, synthetic traces whose medians the predictor was given, 8.1 ms margin on the full-load-window p95, 3 of 7 full-load buckets over the gate, N=400 collapsed. The restore-vs-re-prefill physics is PROVEN single-request but is LMCache's mechanism, and the same repo holds CONTRADICTED results (0.37x single-request, P99 1.6-2.2x worse under concurrency on a disk tier). The agent regime that the brief's larger customer and pricing depend on has zero in-engine evidence above 16k tokens (vLLM crashed at >=32k), the cost model is HF-level and modeled, and tiering enforcement does not exist in code. Nothing has been reproduced across seeds, GPUs, engines or real traces, so this is 'measured but bounded/contradicted', not B.

**Weeks to first credible public proof:** 8-10 weeks to a credible public Metric A (voice) proof; 12-14 weeks for Metric B (agent). Assumptions: 1-2 engineers; the voice connector port from vLLM 0.9.2 to 0.28 (changed KVConnector hooks and CachedRequestData shape, 3-arg ctor) takes 2-3 weeks including provenance and server-metric capture; 3-seed sweeps at N {250,300,350,400} on L40S and H100 against vLLM OffloadingConnector LRU/ARC, LMCache 0.5.x and FP8 KV take ~100-150 GPU-h (~$300-600 at the $1.95-3.95/GPU-h Modal rates in 05-research-gpu-pricing.md) and 1-2 weeks wall-clock; 1 week write-up. Synthetic traces are acceptable for v1 only if the write-up says so; transcript-derived traces depend on a design partner. Metric B is gated on something never demonstrated in either repo: stable in-engine restore at 32k-128k on a native-128k model, plus RAM-tier restore under concurrency, so add 4+ weeks and treat the date as unknown until the week-4 in-engine test passes.

---

## 1. Thesis

A session-aware serving layer for open-model inference: a residency scheduler running inside vLLM through the stock KVConnector hooks, plus a session router and a small control plane, deciding per session, from a return-time hint, context size and per-tier break-evens, whether a session's KV stays pinned in HBM, moves to pinned RAM or NVMe, or is dropped and re-prefilled. Every shipping KV-cache layer evicts with an LRU-family policy that does not know when a session will return (LMCache, Mooncake, vLLM OffloadingConnector, AIBrix; 05-research-kvcache.md), and the hint protocols that would tell it (SGLang RFCs #24656, inactive, and #27574, open) are unimplemented. The repos hold one in-engine case where a return-time hint beat the reactive baseline (voice arm D: 300 vs 200 sessions per L40S inside a 300 ms p95 TTFT gate; one seed; 8.1 ms margin on the full-load window) and one where pinned-RAM restore beat re-prefill 10-17x single-request (dexa, vLLM+LMCache). Sold as a hosted session-addressed endpoint and as BYOC software plus Wafer-style optimization engagements (05-research-wafer.md). The customer sentence: "It holds our sessions warm exactly as long as it pays to, so TTFT does not fall off a cliff when concurrency goes up."

Stated up front: the differentiating claim (hint-driven residency beats LRU at matched memory) is measured once, on synthetic traces, on a 46 GB GPU, with a predictor that used the trace's own medians. Everything else measured is LMCache's restore physics.

## 2. Customer and workload

Two buyers, both already on vLLM or SGLang.

Voice and agent platforms that self-host the LLM. LiveKit serves Gemma 4 31B on SGLang at 192 ms TTFT for $0.40/$1.20 per 1M tokens; ElevenLabs hosts Qwen3.6-35B-A3B and Qwen3.5-397B-A17B; Inworld serves Gemma 4 26B (05-research-voice.md). All but Bland accept a bring-your-own OpenAI-compatible endpoint. Workload as modeled in the voice repo: 7B model, ~2.5k tokens per session (~140 MB KV at 57,344 B/token fp16), 6-30 turns, idle-window p50 24.0 s, model idle 95.2% of wall-clock (vi-phase0-duty-cycle). Caveat: every timing distribution is assumed; phase 0 used 131,072 B/token and 60 tok/s decode, not the served 57,344 B/token and measured ~45 tok/s; no real transcript has been replayed. They pay by concurrency: Vapi $10/line/month beyond 10, Retell $8/concurrency/month beyond 20, ElevenLabs 2x burst pricing.

Agent builders and the providers serving them. TraceLab's ~4,300 Claude Code/Codex sessions: median prefix 115,584-126,180 tokens per step, ~857-886 new input tokens, 184-252 output, human gaps 1.4 min median / 20.6 min p90, prefix-cache hit 95.7% overall but 84.4% on user-initiated steps (05-research-agents.md). llm-d's 219 production Claude Code sessions: median 195K input / 317 output, 96% of requests reuse >=90% of input verbatim, inter-turn pauses 2 s median / 11.4 min p99 (05-research-kvcache.md). So hour-scale idle is the tail, not the median; SJTU/Alibaba traces put 80% of reuses within 10 minutes (Trace A) or 10 seconds (Trace B). They pay per token with cache-read discounts (Fireworks 50%; Anthropic 0.1x reads, 2x for a 1-hour write) on TTL-bound caches.

## 3. The pain, in the customer's words

- "It doesn't go off a cliff when you increase the requests per minute." Neon Health, on Wafer (05-research-wafer.md). The cliff is measured: baseline vLLM on an L40S went from ~92% prefix-cache hits at N=200 to 1.6-1.9% at N=300 and collapsed rather than degraded (vi-armA-baseline-knee; single seed; the aborted N=300 run has no p95).
- "Prompt caching only works within 1 replica"; cached prompts stay "at least several minutes... up to several hours." Fireworks docs (05-research-agents.md).
- Codex on Azure past ~150K tokens: "total prompt-cache misses," 3.28M re-billed tokens = 71% of fresh input (openai/codex #25604). 17.1% higher spend from a 1h-to-5m TTL change across 119,866 calls (anthropics/claude-code #46829, community-measured).
- "Memory your engine doesn't know about will kill it." voice-inference docs/FINDINGS.md §4, after connector staging buffers outside vLLM's budget OOM'd the engine. Building this in-house takes "20 engineers and three or four months" (Tensormesh CEO, 05-research-kvcache.md).

## 4. Value proposition and the proof-of-value benchmark

Value: on a fixed GPU, more sessions inside the latency SLA (voice), and cheaper, faster resumes across idle gaps from seconds to hours (agents), with the residency tier returned per request so hits are observed, not inferred.

Metric A (voice), pre-registered: maximum concurrent sessions N with p95 TTFT <= 300 ms over the full-load window (minutes 5-40, not whole-run), one GPU, median of 3 seeds, turns lacking a TTFT counted as failures. Baselines on vLLM 0.28: vanilla prefix caching; OffloadingConnector (LRU, ARC); LMCache 0.5.x; ours without hints; ours with FP8 KV and hints. Measured so far, seed 1 only, L40S 46 GB, Qwen2.5-7B fp16, vLLM 0.9.2, LMCache 0.3.2: vanilla 200; LMCache 200; ours without hints 200 (N=300 p95 316.8 ms whole-run, 336.8 ms full-load window); with predictive pinning 300 (N=300 p95 272.2 ms whole-run, 291.9 ms full-load window; N=400 34,048 ms). No OffloadingConnector, FP8 KV or 80 GB GPU point exists. Ship target: >=2x vanilla on 3 seeds on the L40S, a measured ratio on an H100, and a margin over the best OSS baseline. Kill: <1.3x vs vanilla, or <1.15x vs the best OSS offloading baseline, on 3 seeds.

Metric B (agent): warm-resume TTFT p95 and GPU-seconds per turn on 100K-token-class sessions across an idle ladder {10 s, 2 min, 20 min, 2 h}, concurrency 32, fixed CPU-RAM and NVMe budget, rungs weighted by the measured gap distributions. Baselines: vanilla re-prefill; LMCache LRU; OffloadingConnector ARC; Fireworks and Tensormesh serverless. Measured so far (single request, no ladder, no concurrency): LMCache CPU restore 10.0x at 4k (301 to 30 ms) and 17.1x at 16k (1,320 to 77 ms) on vLLM 0.24.0/A100; in-engine runs never exceeded 16k because vLLM crashed at >=32k in three configs. 100K-token behavior is unmeasured in any engine in either repo. Target: unmeasured. Reference ceiling: SJTU/Alibaba's reuse-probability eviction beat LRU/FIFO/LFU by 1.5-3.9% hit rate and up to 41.4% mean response time. Kill: policy fails to beat LRU by >=15% on TTFT p95 or GPU-seconds at matched memory.

Why a skeptical buyer would believe it: the harness is theirs to run (evidence-grade flag, per-point archive, seed control, full-load-window p95); the baselines are the OSS configs they already run; the residency tier comes from the connector; the repos' losses (section 6) stay in the pitch.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| Engine (vLLM 0.28.0; SGLang 0.5.18 later) | OSS, unmodified | Prefill, decode, paged HBM KV, prefix caching; HBM eviction order stays vLLM's |
| Session-aware connector | Custom (voice_kv_connector.py 574 lines at vLLM 0.9.2; dexa vllm_connector.py 917 lines at 0.24.0) | Save prompt KV at prefill, restore on match, delay-free (pin) on hint, evict its own RAM slab by policy |
| Residency policy engine | Custom, new (seeded by sessions/tiering.py, 111 lines, advisory today) | Hinted return x context x tier break-even -> pin / RAM / NVMe / drop |
| Hint protocol | Custom, published spec | kv_transfer_params.idle_ms today (6-line client function); extend to {expected_return_ms, ttl_ms, priority, workflow_id, step_id} |
| Session router | Custom, new; llm-d EPP plugin form for BYOC | Session-to-replica affinity over a residency index |
| RAM / NVMe tiers | Custom pinned slab or LMCache / OffloadingConnector; NVMe via LMCache disk or vLLM FS tier | Contiguous session-granular transfers; NVMe unmeasured in either repo |
| Gateway + control plane | Custom (692 + 620 lines; 29 tests pass; forces stream=False) | Keys, metering, credit ledger, OpenAI-compatible front door |

Differentiation lives in the policy layer and the hint protocol, surfaced through a session router; not kernels, the model, or the transfer mechanism (LMCache already moves 16,384 tokens at 22.0 GB/s single-request). The stock KVConnector API can express save-at-prefill, restore-on-match and delay-free, but not HBM eviction ordering or request-less prefetch (voice docs/FINDINGS.md §3.2): the product replaces the free-on-finish decision and the RAM-slab eviction order; vLLM's HBM LRU stays. Proactive turn-end offload was never built. The API has been "experimental" since April 2025; the 0.9.2 to 0.28.0 port is real work. SGLang's HiCacheStorage (get/exist/set) has no pin hook; that path waits on RFC #27574.

```
 orchestrator (voice / coding agent / platform)
     | request + hints {session_id, expected_return_ms, ttl_ms, priority}
     v
 [Session Router]  session->replica affinity; reads residency index
     v
 [vLLM] --KVConnector hooks-- [Session-aware connector]
     | HBM paged KV (vLLM LRU;             | save / restore / delay-free
     |  connector can only delay free)     v
     |                        [Residency policy engine]
     |                  hinted return x ctx size x tier $/GB-hr
     |                        -> pin | RAM | NVMe | drop
     v                                      v
 response {residency: hbm|ram|nvme|cold}   [pinned RAM slab] -> [NVMe] (unmeasured)
```

## 6. Evidence

### Proven

- Baseline collapse is abrupt on a 46 GB GPU: ~92% prefix-cache hits at N=200 to 1.6-1.9% at N=300; ~42 GB working set vs ~26 GB pool (doc estimate). vi-armA-baseline-knee; single seed.
- Pinned-RAM restore beats cold prefill in a real vLLM stack, single request: LMCache 10.0x at 4k, 17.1x at 16k; 0.875 GB in 39.8 ms. lmcache-offload-restore (LMCache version unpinned). LMCache's mechanism, not the product's.
- Cross-instance KV resume is bit-identical (TP=1, small models); 10/10 hook signatures match vLLM 0.24.0.
- fp8/int8 KV at 0.5x bytes: 100% greedy agreement over 48 tokens (one passage, HF-level). Harness: 287,015 measured turns across 19 archived GPU runs.

### Bounded / contradicted

- Arm D (moved from the draft's Proven list; ledger status BOUNDED): one seed, one GPU, one passing N; margin 27.8 ms whole-run, 8.1 ms full-load-window; 3 of 7 full-load buckets over 300 ms; N=400 failed at 34,048 ms with the 10 GB budget; the hint uses the trace's own medians; server-side hit rates unarchived.
- The connector holding N=300 stable where LMCache 0.3.2 collapsed (316.8 vs 25,997.7 / 6,025.1 ms layerwise) is BOUNDED: not a controlled comparison, two LMCache configs, no transfer timings archived, the "7 ms restore" is a docstring estimate, and the tail ratio is stable-vs-collapsed.
- HF-level restore curve (11.8x at 4k to 28.9x at 64k; NVMe 246/390/646/1,228 ms): HF eager baseline 5-7x slower than vLLM prefill, naive torch.load for NVMe, single request; restore is sub-linear in bytes, so part of the widening is fixed-overhead amortization.
- Restore vs re-prefill is contradicted inside dexa: 10-34x (Qwen2.5-7B, pinned RAM, single request) versus 0.37x at 8B/8k and P99 1.6-2.2x worse under 16-24-way concurrency (Llama-3.1-8B, disk-backed connector at ~0.6-1 GB/s). No run measures RAM-tier restore of 16k-64k sessions under concurrency.
- The residency cost model (6.0x/4.9x/2.0x cheaper at 2/15/120 min idle; 64k break-evens HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr; assumed GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr) uses HF prefill times, which inflate the re-prefill side; "2-6x cheaper" is not measured.
- Nothing enforces tiering: tiering.py emits an advisory label; warm is inferred as not-first-touch; the 3.7x demo is one run with no artifact.
- Arm C N=300 (14,958 ms): server died at minute 41, 37% of turns lack a TTFT; collapse predates the outage.
- Connector limits: TP=1, fp16/bf16, non-MLA, prompt-KV only, synchronous loads inside the scheduler step, pinned to vLLM 0.9.2; a store evicts the entry it matched, so sessions sharing a system prompt thrash each other's slab entries (unquantified). Correctness suite unarchived.

### Unproven

| Claim | Experiment that proves or kills it | Est. cost (author's estimate) |
|---|---|---|
| Arm D survives seeds, a modern baseline, FP8 KV, an 80 GB GPU and a buyer-sized model | Seeds 1-3, N {250,300,350,400}, vLLM 0.28: vanilla, OffloadingConnector LRU/ARC, LMCache 0.5.x, ours +/- hints, +/- FP8 KV; L40S and H100; one 26-35B point | ~40 GPU-h per GPU class; 4 days each |
| A real predictor matches trace-median accuracy | Transcript-derived traces; hint error, pin hit rate, wasted-pin bytes | partner data; ~8 GPU-h |
| vLLM restores 32k-128k in-engine at all, then session-aware beats LRU in the agent regime | Native-128k model with chunked prefill; then the Metric B ladder at C=32 vs LMCache-LRU and OffloadingConnector-ARC | ~10 GPU-h, 2 days; then ~40 GPU-h H100, 7 days |
| RAM-tier restore holds under concurrency at 16k-64k; NVMe fast enough for the modeled 4.9 hr break-even; $ per session-hour at invoice rates | Crossover surface (context x concurrency x tier) with LMCache disk / vLLM FS tier, metered GPU-seconds and bytes | ~40 GPU-h; 7 days |
| Turn-end offload bounds the working set; restore correctness; SGLang path; TP>1 and chunked prefill | Free-after-save at N=300/400; archive modal_kv_correctness.py; port onto HiCache once a pin hook lands; 64k resume | ~8, ~1, unscoped, ~15 GPU-h |

## 7. MVP and 6-week build plan

Ships first: an evidence-grade benchmark report plus a BYOC package (connector + policy engine + harness), then a hosted session-addressed endpoint on the same stack. Reused: voice-inference vkv/backends/voice_kv_connector.py; vkv/run.py _idle_hint; the vkv harness (orchestrator, loadgen, metrics/analyze.py, evals/modal_arm_a.py); dexa src/dexa/engine/vllm_connector.py (async load); dexa_platform gateway/control/sessions; evals/stateful_cost_model.py; evals/modal_lmcache_restore.py and modal_vllm_warmstart.py (baselines). New: enforced policy, hint spec, router, NVMe, multi-seed aggregation, server-metric capture into events.jsonl, manifest provenance.

- Weeks 1-2: Port the voice connector to vLLM 0.28 KVConnectorBase_V1 (hooks and the CachedRequestData shape changed since 0.9.2); merge dexa's async load; add provenance and server-metric scraping; archive modal_kv_correctness.py output. The step most likely to slip.
- Week 3: Metric A at seeds 1-3, N {250,300,350,400}, L40S and H100, against OffloadingConnector and LMCache 0.5.x, with and without FP8 KV. First kill gate.
- Week 4: Policy engine v1: replace advisory tiering.py with an enforced decide() returning pin/free/demote at request_finished; residency tier in response headers read from the connector. In parallel, get 32k-128k restore working in-engine on a native-128k model.
- Week 5: Metric B ladder vs LMCache-LRU and OffloadingConnector-ARC; NVMe measured for the first time. Second kill gate.
- Week 6: Router (session affinity, residency index); hosted endpoint on Modal behind the gateway with streaming; public write-up with raw logs and BYOC docker-compose. The llm-d EPP plugin and SGLang path fall after week 6.

Realistic slip: if the port takes three weeks or 100K in-engine fails, Metric B lands in weeks 8-9 and the public proof is Metric A only.

## 8. Pricing model

Three SKUs the architecture can price because the policy engine computes tier, bytes and break-even per session:

1. Hosted endpoint: per-token at a cache-read discount plus a residency line billed per session-hour by tier (precedents: Anthropic Managed Agents $0.08/session-hour while running; Gemini explicit-cache storage $0.50-$4.50 per 1M tokens per hour; DeepInfra 5m/1h retention at 1.25x/2x write). The customer sets ttl_ms; a held session costs bytes x tier rate x hours. The cost model's rates are assumed, not invoice rates (05-research-gpu-pricing.md).
2. Voice/agent platforms (BYOC or dedicated): a per-GPU-hour license priced against sessions-per-GPU, the unit they sell (Vapi $10/line/month; Retell $8/concurrency/month). Wafer's dedicated endpoints are custom-priced; Tensormesh's post-v1 formula is 30% of estimated savings.
3. Optimization engagements: fixed-fee benchmark plus integration, the Wafer motion (its YC-era contracts saved "$5k/day once deployed").

Per-token providers do not express this today: their cache is a shared LRU with a TTL (Fireworks "several minutes... up to several hours"; xAI "evicted at any time"; OpenAI prompt_cache_key "does not pin"). The closest analogues are TTL-priced retention (Anthropic 1h at 2x write; DeepInfra 5m/1h), not a per-session, byte-metered residency hour.

## 9. Competitive facts

| Who | Adjacent shipping | Not shipping (per research files) | Source |
|---|---|---|---|
| LMCache / Tensormesh | CPU/disk/S3/NIXL tiers; LRU-family eviction; SaaS with $0 cached input; $24.5M raised; Operator "coming soon"; post-v1 price 30% of estimated savings | TTL- or idle-window-aware eviction; third-party engine support | 05-research-kvcache.md |
| Mooncake Store | Approximate LRU; 10 s leases; soft pins (30 min default) and hard pins; SSD offload; draft PR #2835 TTL-bounded leases | Draft is experimental; DFS tier "not production-ready" | 05-research-kvcache.md |
| NVIDIA Dynamo KVBM | G1-G4 tiers; presence/LFU offload filters; cache_control-style TTL retention described in a blog | Per-tier eviction algorithm; performance numbers | 05-research-kvcache.md; 05-research-agents.md |
| llm-d | Precise KV-event prefix routing; OffloadingConnector LRU/ARC; P2P KV pull; session affinity; powers GKE Inference Gateway | Predicted-return or TTL policy; NVMe "inactive" in its Claude Code measurements | 05-research-kvcache.md |
| vLLM | OffloadingConnector (lru/arc, custom policy hook); RFC #38260 TieringManager open; KVConnector V1 "experimental" | Session-aware policy in tree | 05-research-kvcache.md |
| FlexKV (Tencent) | In vLLM 0.17.2+ and SGLang 0.5.16+; "logical LRU" with frequency-aware grace time | Hint protocol | 05-research-kvcache.md |
| SGLang | HiCache L1/L2/L3; RFC #24656 (open, inactive); RFC #27574 pin/demote/prefetch/TTL (open); roadmap Q3 "session-aware RadixTree" | L2 eviction policy documentation; hints | 05-research-kvcache.md |
| Fireworks; DeepInfra; Baseten | Per-replica cache with session-affinity headers, 50% cached discount; DeepInfra 5m/1h retention at 1.25x/2x write; Baseten KV-aware routing | Guaranteed retention; byte-metered residency | 05-research-agents.md; 05-research-providers.md |
| Morph | Sticky KV placement via x-session-id; per-request TTL control; Reflexes engine "forked from vLLM"; self-reported ~$7M run rate | Open-model residency as a product; BYOC | 05-research-morph.md |
| Wafer | Profile/explore/deploy agents across engines and hardware; dedicated endpoints; $40M Series A; Neon Health TTFT 800 to ~550 ms | Session-level KV residency policy on its technology page | 05-research-wafer.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| Arm D 1.5x is seed noise (8.1 ms margin) | 3-seed median max-passing N, full-load-window p95 | <1.3x vs vanilla | >=1.5x, spread <0.2 |
| A modern reactive baseline erases the hint delta | OffloadingConnector LRU/ARC and LMCache 0.5.x, same sweep | Best baseline passes N=300 and hints add <15% N | Hints add >=25% N |
| FP8 KV moves the wall for everyone | Same sweep with fp8 KV on vanilla and ours | Vanilla+fp8 passes N=400 and hints add <10% | Hints add >=25% at matched precision |
| 80 GB HBM dissolves the voice wall | Same sweep on H100; then a 26-35B point | Vanilla holds >=2x the voice-realistic N and policy adds <10% | Policy adds >=25% sessions or cuts GPU-h/session >=25% |
| Predictor accuracy on real calls | Pin hit rate, wasted-pin bytes on transcript-derived traces | Hit <60% or waste >40% of budget | Hit >=80% |
| Agent regime: 128k never works in-engine, or policy fails to beat LRU | 32k/64k/128k restore on a native-128k model; Metric B vs LMCache-LRU at matched memory | No stable 128k by week 4; <15% on both metrics | Stable, archived; >=25% on either |
| RAM-tier restore stalls under concurrency | Crossover surface at 16k-64k, C=32 | Policy p99 worse than re-prefill at any rung | Async load keeps p99 within 1.2x |
| NVMe too slow; cost model wrong at invoice rates | Restore ms at 64k vs modeled 1,228 ms; metered $/session-hour | >3x modeled; never cheaper at idle <=20 min | Within 2x; cheaper at 2-20 min for >=16k |
| Silent KV corruption; shared-prefix slab thrash | modal_kv_correctness.py; hit rate at 5 vs 300 distinct system prompts | Any missing-entry load; >20-point drop | 0 errors archived; within 5 points |

## 11. Founder decisions

- Hosted vs BYOC vs OSS-core: hosted first (dexa's direction), BYOC first (the voice repo's shape), or open-source the connector and sell policy/router/support. Evidence: which design partner converts.
- Voice-first vs agent-first. Voice has the only in-engine win (1.5x, one seed) and concurrency-priced buyers; agents have larger contexts and the RFC-documented gap, but no in-engine result above 16k.
- Own connector vs a cache_policy_module_path plugin on vLLM's native OffloadingConnector (less control over transfer mechanics). Evidence: the week-3 head-to-head.
- Whether Tensormesh's, Dynamo's, llm-d's or SGLang's adjacency matters (section 9); the RFCs could land any quarter.
- Model and GPU class for the proof: 7B fp16 on L40S exists; buyers run 26-35B models on 80 GB+ parts.
- The SLA gate: 300 ms p95 TTFT is the voice PRD's number; Twilio budgets 375 ms target / 750 ms max. A looser gate changes every N in section 4.
- The ship threshold: the voice README's table put 1.5x in its "<2x kill" band; RESULTS.md re-scored it "above kill, below proceed." Which governs is a founder call.
- vLLM-only for six weeks, or vLLM plus SGLang; and whether the hint protocol is an open spec (what RFC #27574 could adopt) or proprietary. LiveKit runs SGLang; the repos have zero SGLang code.
- NVMe now, or after the RAM-tier win is banked; the 4.9 hr break-even is modeled only.
- Whether to pitch the hour-scale idle regime (the tail of measured gap distributions) or the 2 s-20 min regime (the bulk); this changes the SKU and the baseline.
- Whether the open-model voice/agent serving market is large enough. In the files: Vapi 1-5M calls/day, ElevenLabs 10M+ conversations/week; a self-hosted LLM is under 5% of a voice minute (Inworld) or ~30% (smallest.ai).

## 12. Combinations

- Voice-specific serving (arm D productized): this brief is its generalization; the voice product is Metric A with a voice-shaped policy on the same connector.
- Stateful agent-session provider (dexa's STATEFUL_SESSIONS): the hosted SKU here; this brief supplies the enforcement it lacks.
- Wafer-style optimization engagements: the harness and connector are the deliverables; residency is one knob the loop tunes.
- Evidence-grade benchmark product: the vkv harness plus server-metric capture is sellable alone and is the weeks-1-2 deliverable regardless.
- Computer-use or document-VLM gateways: same router and control plane; the ~2x delta-perception result (bounded) composes with residency without depending on it.
