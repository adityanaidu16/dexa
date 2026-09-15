## 1. Thesis

A session-aware serving layer for open-model inference: a residency scheduler that runs inside vLLM/SGLang through the stock KV-connector hooks, plus a session router and a small control plane, and decides per session, from predicted return time, context size and per-tier break-evens, whether that session's KV cache stays in HBM, moves to pinned RAM or NVMe, or is dropped and re-prefilled. Every shipping KV-cache layer evicts with an LRU-family policy that does not know when a session will return (LMCache LRU/IsolatedLRU, Mooncake approximate LRU, vLLM OffloadingConnector LRU/ARC, AIBrix S3FIFO/LRU/FIFO; 05-research-kvcache.md), and the hint protocols that would tell it (SGLang RFCs #24656 and #27574) are open or inactive. The two repos contain one measured case where a return-time hint beat LRU (voice arm D: 300 vs 200 sessions per L40S inside a 300 ms p95 TTFT gate) and one where RAM restore beat re-prefill 10-17x single-request (dexa, vLLM+LMCache). Sold two ways: a hosted session-addressed endpoint for agent and voice builders, and software plus optimization engagements for inference providers and voice/agent platforms, the Wafer-style "optimize the stack for your workload" motion (05-research-wafer.md). The sentence a customer repeats: "It holds our sessions warm exactly as long as it pays to, so TTFT does not fall off a cliff when concurrency goes up."

## 2. Customer and workload

Two buyers, both already running open models on vLLM or SGLang.

Voice and agent platforms that self-host the LLM. LiveKit serves Gemma 4 31B on SGLang at 192 ms TTFT for $0.40/$1.20 per 1M tokens; ElevenLabs hosts Qwen3.6-35B-A3B and Qwen3.5-397B-A17B; Inworld serves Gemma 4 26B; Deepgram offers Nemotron-3-nano-30B (05-research-voice.md). Every orchestrator fetched except Bland accepts a bring-your-own OpenAI-compatible endpoint. The workload as modeled in the voice repo: 7B-class model, ~2.5k tokens per session (~140 MB KV at 57,344 B/token fp16), 6-30 turns, gaps dominated by TTS playback, idle-window p50 24.0 s, model idle 95.2% of wall-clock (02-evidence-ledger-voice.md, vi-phase0-duty-cycle; synthetic distributions). They pay by concurrency: Vapi $10/line/month beyond 10, Retell $8/concurrency/month beyond 20, ElevenLabs 2x burst pricing.

Agent builders and the providers serving them. TraceLab's ~4,300 Claude Code/Codex sessions: median prefix 115,584-126,180 tokens per step, ~857-886 new input tokens, 184-252 output, human gaps 1.4 min median / 20.6 min p90 (05-research-agents.md). llm-d's 219 production Claude Code sessions: median 195K input / 317 output, 96% of requests reuse >=90% of input verbatim, inter-turn pauses 2 s median / 11.4 min p99 (05-research-kvcache.md). OpenHands recommends Qwen3.6-35B-A3B on vLLM/SGLang; open-weight models carried a majority of OpenRouter tokens by mid-2026, concentrated in coding and agentic work. They pay per token with cache-read discounts (Fireworks 50%; Anthropic 0.1x reads, 2x for a 1-hour write) on caches that are per-replica and TTL-bound.

## 3. The pain, in the customer's words

- "It doesn't go off a cliff when you increase the requests per minute." Neon Health, on choosing Wafer (05-research-wafer.md). The cliff is measured: baseline vLLM on an L40S went from a 92% prefix-cache hit rate at N=200 to 1.6-1.9% at N=300 and collapsed rather than degraded (02-evidence-ledger-voice.md, vi-armA-baseline-knee).
- "[We] reserve more headroom than a throughput-maximized deployment would." LiveKit on serving Gemma 4 for voice (05-research-voice.md). Headroom is paid for in GPUs.
- "Prompt caching only works within 1 replica"; cached prompts stay "at least several minutes... up to several hours." Fireworks docs (05-research-agents.md).
- Codex on Azure past ~150K tokens: "total prompt-cache misses," 3.28M re-billed tokens = 71% of fresh input (openai/codex #25604). 17.1% higher spend from a 1h-to-5m TTL change across 119,866 calls (anthropics/claude-code #46829).
- "Orchestrators know session structure and tool-gap duration that request-local LRU cannot observe." SGLang RFC #27574 (05-research-kvcache.md).
- "Memory your engine doesn't know about will kill it." voice-inference docs/FINDINGS.md §4, after connector staging buffers outside vLLM's budget OOM'd the engine. Building this in-house takes "20 engineers and three or four months" (Tensormesh CEO, 05-research-kvcache.md).

## 4. Value proposition and the proof-of-value benchmark

Value: on a fixed GPU, more sessions inside the latency SLA (voice regime), and cheaper, faster resumes across idle gaps from seconds to hours (agent regime), with the residency decision returned per request so a buyer sees hits rather than infers them.

Metric A (voice regime), pre-registered: maximum concurrent sessions N with p95 TTFT <= 300 ms over the full-load window (minutes 5-40, not whole-run), one GPU, median of 3 seeds. Setup: the voice repo's harness (vkv, 287,015 archived measured turns) with real transcript-derived traces replacing the synthetic distributions. Baselines: vanilla vLLM with prefix caching; vLLM + LMCache (default and layerwise); vLLM OffloadingConnector (LRU and ARC); SGLang HiCache. Measured so far, seed 1, L40S 46 GB, Qwen2.5-7B fp16, vLLM 0.9.2: vanilla 200; LMCache 200 (N=300 p95 25,997.7 ms default, 6,025.1 ms layerwise); our connector without hints 200 (N=300 p95 316.8 ms, stable but 16.8 ms over the gate); with predictive pinning 300 (N=300 p95 272.2 ms whole-run, 291.9 ms full-load window; N=400 34,048 ms). Ship target: >=2x baseline on 3 seeds on the L40S and a measured ratio on an 80 GB GPU. Kill: <1.3x on 3 seeds.

Metric B (agent regime): warm-resume TTFT p95 and GPU-seconds per turn on 100K-token-class sessions across an idle ladder {10 s, 2 min, 20 min, 2 h}, concurrency 32, fixed CPU-RAM and NVMe budget. Baselines: vanilla vLLM re-prefill; vLLM + LMCache LRU; vLLM OffloadingConnector ARC; the same trace against Fireworks and Tensormesh serverless endpoints. Measured so far (single request, no idle ladder): LMCache CPU restore 10.0x at 4k (301 to 30 ms) and 17.1x at 16k (1,320 to 77 ms) on vLLM 0.24.0/A100; in-engine runs never exceeded 16k. Target: unmeasured. Ceiling reference: the SJTU/Alibaba trace study, where reuse-probability eviction beat LRU/FIFO/LFU by 1.5-3.9% hit rate and up to 41.4% mean response time. Kill: policy fails to beat LRU by >=15% on TTFT p95 or GPU-seconds at matched memory.

Why a skeptical buyer would believe it: the harness is theirs to run (evidence-grade flag, per-point archive, seed control, full-load-window p95); the baselines are the OSS configs they already run; the residency tier is returned per request, so a hit is observable. The repos' history is the honesty argument: dexa's own connector lost to vLLM re-prefill (0.37x at 8B/8k; P99 1.6-2.2x worse under 16-24-way concurrency) even after load-path optimization, and only the pinned-RAM path (LMCache) won; voice arm C's audio-triggered prefetch collapsed at N=300. Both stay in the pitch.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| Engine (vLLM 0.28.0 / SGLang 0.5.18) | OSS, unmodified | Prefill, decode, paged HBM KV, prefix caching |
| Session-aware connector | Custom, on KVConnectorBase_V1 (from voice_kv_connector.py 574 lines; async load and adaptive policy from dexa vllm_connector.py 917 lines) | Save prompt KV at prefill, restore on match, delay-free (pin) on hint, demote on policy |
| Residency policy engine | Custom (seeded by dexa_platform/sessions/tiering.py 111 lines and evals/stateful_cost_model.py) | Per session: predicted return x context size x tier break-even -> HBM / RAM / NVMe / drop; enforced, not advisory |
| Hint protocol | Custom, published spec | kv_transfer_params.idle_ms today; extend to {expected_return_ms, ttl_ms, priority, workflow_id, step_id}, aligned to SGLang RFC #27574 vocabulary (pin/demote/prefetch/TTL) |
| Session router | Custom; llm-d EPP plugin form for BYOC | Session-to-replica affinity that reads residency state; precise index from vLLM KVEvents (OSS) |
| RAM tier | Custom pinned slab (voice connector) or LMCache / OffloadingConnector | Contiguous session-granular transfers |
| NVMe tier | OSS (LMCache local disk, vLLM FS tier, Mooncake SSD offload) | Unmeasured in either repo; see Unproven |
| Benchmark harness | Custom (vkv traces, orchestrator, loadgen, metrics; ~2,000 lines) | Evidence-grade multi-turn session load with full-load-window p95 |
| Gateway + control plane | Custom (dexa_platform gateway 692 lines, control 620 lines; 29 tests pass) | Keys, metering, credit ledger, OpenAI-compatible front door |

Where the differentiation lives: the policy layer and the hint protocol between orchestrator and engine, surfaced through a session router. Not kernels, not the model, not the transfer mechanism (LMCache already moves 16,384 tokens at 22.0 GB/s). vLLM and SGLang stay unmodified: the connector uses the stock hooks (10/10 signature match on vLLM 0.24.0; the API has been labeled "experimental and subject to change" since April 2025, a maintenance cost rather than a fork). Replaced: LRU eviction ordering and the free-on-finish decision. Added: return-time hints, per-tier break-even, session-to-replica affinity.

```
 orchestrator (voice / coding agent / platform)
     | request + hints {session_id, expected_return_ms, ttl_ms, priority}
     v
 [Session Router]  session->replica affinity; reads residency index
     v
 [vLLM / SGLang] --KVConnector hooks-- [Session-aware connector]
     | HBM paged KV (pin/free)              | save / restore / demote
     |                                      v
     |                        [Residency policy engine]
     |                  predicted return x ctx size x tier $/GB-hr
     |                        -> HBM | RAM | NVMe | drop
     v                                      v
 response {residency: hbm|ram|nvme|cold}   [pinned RAM slab] -> [NVMe] (unmeasured)
```

## 6. Evidence

### Proven

- Predictive pinning raised the 300 ms-p95 ceiling from 200 to 300 sessions on one L40S (arm D N=300 p95 272.2 ms; A/B/C max passing 200), using response length plus turn-taking medians, no audio events. 02-evidence-ledger-voice.md, vi-armD-predictive-pinning; runs/modal/sweep_d.
- Baseline collapse is abrupt: prefix-cache hit rate 92% at N=200 to 1.6-1.9% at N=300; ~42 GB working set vs ~26 GB pool. vi-armA-baseline-knee.
- A 574-line stock-API connector held N=300 stable (p95 316.8 ms) where LMCache 0.3.2 collapsed (25,997.7 ms default; 6,025.1 ms layerwise). vi-armB-voice-connector-reactive.
- RAM restore beats cold prefill in a real vLLM stack: LMCache 10.0x at 4k, 17.1x at 16k; 0.875 GB moved in 39.8 ms. 01-evidence-ledger-dexa.md, lmcache-offload-restore.
- In-GPU prefix hit vs cold: ~12x at 4k, 25-34x at 16k on vLLM 0.24.0. vllm-warmstart-prefix-cache.
- HF-level restore curve widens with context: 11.8x at 4k to 28.9x at 64k; NVMe restore 246/390/646/1,228 ms at 4k/16k/32k/64k. stateful-warm-session-hf.
- KV saved by one vLLM process loads bit-identically in another (TP=1); 10/10 connector hooks conform to vLLM 0.24.0. cross-instance-resume; connector-conformance.
- fp8/int8 KV at 0.5x bytes gives 100% greedy agreement over 48 tokens; int4 does not (29%). kv-interchange-formats.
- Harness: 287,015 measured turns across 19 archived GPU runs; 27 tests pass with pytest-asyncio. 03-build-inventory.md.

### Bounded / contradicted

- Arm D's 1.5x is one seed, one GPU, one passing N; margin 27.8 ms whole-run and 8.1 ms full-load-window; 3 of 7 full-load buckets exceeded 300 ms; N=400 failed at 34,048 ms with the 10 GB budget; the hint uses the trace's own median constants, so predictor accuracy on real calls is untested. vi-armD-predictive-pinning; vi-armD-n400-failure.
- Phase-0 95.2% idle is synthetic and computed at 60 tok/s decode while GPU runs measured ~45 tok/s. vi-phase0-duty-cycle.
- Restore vs re-prefill is contradicted inside dexa: 10-34x (Qwen2.5-7B, pinned RAM / LMCache, single request) versus 0.37x single-request and P99 1.6-2.2x worse under concurrency (Llama-3.1-8B, Dexa connector loading from a Modal Volume at ~1 GB/s). Different models, media and load paths; no run measures LMCache under concurrency. restore-vs-reprefill-cross-repo-contradiction.
- The voice connector's 82x/19x tail advantage over LMCache is stable-vs-collapsed, not per-transfer; the "7 ms restore" is a docstring estimate. vi-contiguous-vs-chunked-transfer.
- The residency cost model (6.0x/4.9x/2.0x cheaper at 2/15/120 min idle; 64k break-evens HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr) is modeled from HF prefill times at assumed rates (GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr); dexa_platform/README.md presents "2-6x cheaper" as measured. It is not. residency-cost-model.
- Nothing enforces tiering: tiering.py and edge/tiering.ts emit an advisory label; no code demotes or pins KV in LMCache; vllm_lmcache_backend.py configures a 40 GB CPU tier and no disk. 03-build-inventory.md.
- The end-to-end session demo (12k tokens, 5.4 s to 1.5 s, 3.7x) is one run with no artifact. stateful-session-service-live.
- Arm C N=300 (14,958 ms) has the server unreachable from minute 41 and 37% of turns without TTFT; the collapse predates the outage. vi-armC-n300-archived-run-integrity.
- Both repos name the same top threat: an 80 GB GPU moves the baseline wall past voice-realistic session counts (vi-open-larger-hbm-gpu); llm-d's Claude Code measurements ran with NVMe "inactive."

### Unproven

| Claim | Experiment that proves or kills it | Est. cost |
|---|---|---|
| Session-aware residency beats LRU in the agent regime (100K+ contexts, minute-to-hour gaps) | Replay TraceLab/llm-d-shaped traces at C=32 on vLLM 0.28, LMCache-LRU vs policy engine; TTFT p95 and GPU-s/turn per idle rung | ~40 GPU-h H100; 7 days |
| NVMe restore is fast enough for the ~4.9 hr break-even to be real | Enable LMCache local disk / vLLM FS tier; restore ms at 16k-128k vs modeled 246-1,228 ms | ~10 GPU-h; 2 days |
| $ savings per session-hour ("2-6x cheaper") | Same ladder with metered GPU-seconds and storage bytes at invoice rates | included above |
| Arm D reaches >=2x with tuning (16 GB budget, hold multiplier, N=350) on 3 seeds | Config-only sweep, seeds 1-3, N in {250,300,350,400} | ~24 GPU-h L40S; 3 days |
| The gain survives an 80 GB GPU | Same sweep with --gpu H100 | ~24 GPU-h; 3 days |
| A real predictor matches trace-median accuracy | Convert real transcripts to SessionTrace JSONL; hint error and pin hit rate | partner data; ~8 GPU-h |
| Restore correctness under eviction pressure | Run and archive evals/modal_kv_correctness.py | ~1 GPU-h |
| SGLang path via HiCacheStorage / RFC #27574 hints | Port connector semantics; reproduce Metric A on SGLang | ~30 GPU-h; 10 days |
| Chunked prefill and TP>1 in the connector | Implement; rerun 64k resume bench | ~15 GPU-h; 7 days |

## 7. MVP and 6-week build plan

Ships first: an evidence-grade benchmark report plus a BYOC package (connector + policy engine + harness) a voice or agent platform can run on its own vLLM, and a hosted session-addressed endpoint on the same stack. Reused: voice-inference vkv/backends/voice_kv_connector.py (pin/restore/pinned slab); vkv/run.py _idle_hint (6-line client half of the protocol); the vkv harness (orchestrator, loadgen, metrics/analyze.py, evals/modal_arm_a.py sweep driver); dexa src/dexa/engine/vllm_connector.py (async load, adaptive load-vs-recompute); dexa_platform gateway/control/sessions; evals/stateful_cost_model.py and sessions/tiering.py (break-even table); evals/modal_lmcache_restore.py and modal_vllm_warmstart.py (baselines). New: enforced tiering, hint spec, router, NVMe, multi-seed aggregation, server-metric capture into events.jsonl.

- Week 1: Port the voice connector to vLLM 0.28 KVConnectorBase_V1 (3-arg ctor per dexa's probe); merge dexa's async load; add manifest provenance (git SHA, connector, flags) and server-metric scraping the harness lacks. Archive modal_kv_correctness.py output.
- Week 2: Arm D reproduction at seeds 1-3, N in {250,300,350,400}, L40S and H100. First kill gate.
- Week 3: Policy engine v1: replace advisory tiering.py with an enforced decide() returning pin/free/demote at request_finished from expected_return_ms and per-tier break-even; residency in response headers.
- Week 4: Agent-regime ladder (Metric B) on vLLM 0.28 + LMCache LRU vs policy engine; NVMe tier measured for the first time. Second kill gate.
- Week 5: Router (session affinity, residency index) as a standalone process and an llm-d EPP plugin skeleton; hosted endpoint on Modal behind the dexa_platform gateway with streaming (the gateway currently forces stream=False).
- Week 6: Public benchmark write-up with raw logs; BYOC docker-compose; two design-partner engagements scoped from the report (one voice platform, one agent product).

## 8. Pricing model

Three SKUs the architecture can price because the policy engine already computes tier, bytes and break-even per session:

1. Hosted endpoint: per-token at a cache-read discount plus a residency line billed per session-hour by tier (precedents: Anthropic Managed Agents $0.08/session-hour while running; Gemini explicit-cache storage $0.50-$4.50 per 1M tokens per hour; Tensormesh cached input at $0). The customer sets ttl_ms; a held session costs bytes x tier rate x hours, which the cost model already exposes at assumed rates.
2. Voice/agent platforms (BYOC or dedicated): a per-GPU-hour license priced against sessions-per-GPU, the unit they sell (Vapi $10/line/month; Retell $8/concurrency/month). Wafer's dedicated endpoints are custom-priced; Tensormesh's post-v1 formula is 30% of estimated savings.
3. Optimization engagements for inference providers: fixed-fee benchmark plus integration, the Wafer motion (its YC-era contracts saved "$5k/day once deployed").

Why a per-token provider cannot express this: its cache is a shared LRU with a TTL (Fireworks "several minutes... up to several hours"; xAI "evicted at any time"; OpenAI prompt_cache_key "does not pin"). A residency-hour is a per-session commitment with a known byte cost; the connector's pin/hold and the policy engine's tier decision make it enforceable and meterable.

## 9. Competitive facts

| Who | Adjacent shipping | Not shipping (per research files) | Source |
|---|---|---|---|
| LMCache / Tensormesh | CPU/disk/S3/NIXL tiers; LRU/IsolatedLRU/noop watermark eviction; SaaS with $0 cached input; $24.5M raised; Operator "coming soon"; post-v1 price 30% of estimated savings | No TTL- or idle-window-aware eviction documented; no third-party engine support | 05-research-kvcache.md |
| Mooncake Store | Approximate LRU; 10 s leases; soft pins 30 min default; hard pins; SSD offload; draft PR #2835 TTL-bounded group leases | Draft is experimental; DFS tier "not production-ready" | 05-research-kvcache.md |
| NVIDIA Dynamo KVBM | G1-G4 tiers; presence/LFU offload filters; blog describes cache_control-style TTL retention | Per-tier eviction algorithm undocumented; no performance numbers in the design doc | 05-research-kvcache.md; 05-research-agents.md |
| llm-d | Precise KV-event prefix routing (default); OffloadingConnector LRU/ARC; P2P KV pull; session affinity; powers GKE Inference Gateway | No predicted-return or TTL policy; NVMe "inactive" in its Claude Code measurements | 05-research-kvcache.md |
| vLLM | OffloadingConnector (lru/arc, custom policy hook); RFC #38260 TieringManager open; KVConnector V1 "experimental" | No session-aware policy in tree | 05-research-kvcache.md |
| SGLang | HiCache L1/L2/L3; RFC #24656 agent hints (open, inactive); RFC #27574 pin/demote/prefetch/TTL (open); roadmap Q3 "session-aware RadixTree, KV cache orchestrator" | L2 eviction policy undocumented; hints not implemented | 05-research-kvcache.md |
| Fireworks | Per-replica cache, session-affinity headers, 50% cached discount | Guaranteed retention | 05-research-agents.md |
| Morph | Sticky KV placement via x-session-id; per-request TTL control; vLLM-forked engine; ~$7M run rate, team 3 | Open-model residency as a product; BYOC | 05-research-morph.md |
| Wafer | Profile/explore/deploy agents across engines and hardware; dedicated endpoints; $40M Series A; Neon Health voice TTFT 800 to ~550 ms | Session-level KV residency policy not on its technology page | 05-research-wafer.md |
| Anthropic / OpenAI / Gemini / DeepSeek | 5m/1h TTL at 1.25x/2x write; 30 min-24 h retention; hourly cache storage; disk cache "hours to days" | Per-session pinned KV as a SKU (none found) | 05-research-caching.md |
| Continuum (research) | TTL pinning from reload cost + queueing delay; >8x avg JCT on SWE-Bench/BFCL/OpenHands, 8B-355B models | Not a product; not in any engine | 05-research-kvcache.md |
| Storage vendors (WEKA, VAST, DDN, Pliops, MinIO) | NVMe/flash KV extenders on Dynamo/NIXL; CMX "2H 2026" | Eviction policy layer; pricing undisclosed | 05-research-kvcache.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| Arm D 1.5x is seed noise | 3-seed median max-passing N, L40S, full-load-window p95 | <1.3x vs vanilla | >=1.5x, spread <0.2 |
| 80 GB HBM dissolves the voice wall | Same sweep on H100; N at which vanilla collapses | Vanilla holds >=2x the voice-realistic N and policy adds <10% | Policy adds >=25% sessions or cuts GPU-h/session >=25% |
| Session-aware beats LRU only on synthetic traces | Metric B on TraceLab/llm-d-shaped traces vs LMCache-LRU, matched memory | <15% on both TTFT p95 and GPU-s/turn | >=25% on either |
| NVMe restore too slow to matter | Measured restore ms at 64k vs modeled 1,228 ms | >3x modeled and break-even >12 h | Within 2x of model |
| Predictor accuracy on real calls | Pin hit rate and wasted-pin bytes on transcript-derived traces | Hit <60% or waste >40% of budget | Hit >=80% |
| Connector API drift (vLLM/SGLang) | Port cost per minor release | >1 engineer-week for two consecutive releases | <2 days |
| Restore corrupts KV silently | modal_kv_correctness.py bitwise + forced-evict probe | Any missing-entry load | 0 errors, archived |
| Cost model wrong at invoice rates | Metered $ per session-hour on the ladder | Stateful not cheaper at any idle rung <=20 min | Cheaper at 2-20 min for >=16k contexts |

## 11. Founder decisions

- Hosted vs BYOC vs OSS-core: hosted endpoint first (dexa's FINDINGS direction), BYOC software first (the voice repo's shape), or open-source the connector and sell policy/router/support. Evidence: which week-6 design partner converts; Tensormesh's Operator is "coming soon" while its SaaS is in beta.
- Voice-first vs agent-first. Voice has the only measured win (1.5x, one seed) and buyers priced by concurrency; agents have larger contexts (10-30x restore ratios measured) and the RFC-documented gap, but no measured policy-vs-LRU result. Evidence: Metrics A and B in weeks 2 and 4.
- Whether Tensormesh's, Dynamo's, llm-d's or SGLang's adjacency matters. Facts are in section 9; the RFCs could land any quarter.
- Model and GPU class for the proof. 7B fp16 on L40S is what exists; buyers run 26-35B models and 100K+ contexts on 80 GB+ parts. Evidence: the H100 sweep.
- vLLM-only for six weeks, or vLLM plus SGLang. SGLang has the hint RFCs and LiveKit runs it; the repos have zero SGLang connector code.
- NVMe now, or after the RAM-tier win is banked. The 4.9 hr NVMe break-even is modeled only.
- Whether the open-model voice/agent serving market is large enough. Volumes in the files: Vapi 1-5M calls/day, ElevenLabs 10M+ conversations/week; Inworld puts a self-hosted LLM under 5% of a voice minute, smallest.ai ~30%.
- Whether the hint protocol is an open spec (what SGLang RFC #27574 could adopt) or proprietary.

## 12. Combinations

- Voice-specific serving (arm D productized): this brief is its generalization; the voice product is Metric A with a voice-shaped policy on the same connector.
- Stateful agent-session provider (dexa's STATEFUL_SESSIONS): that is the hosted SKU here; this brief supplies the enforcement it lacks.
- Wafer-style optimization engagements: the harness and connector are the deliverables; residency policy is one knob a profile/explore/deploy loop tunes.
- Evidence-grade benchmark product: the vkv harness plus server-metric capture is sellable alone and is the week-1 deliverable regardless.
- Morph-style specialized models: orthogonal at the model layer; their sessions still need residency, and Morph's sticky placement plus per-request TTL is the closest shipped analogue.
- Computer-use or document-VLM gateways: same router and control plane; the ~2x delta-perception result composes with, but does not depend on, residency.