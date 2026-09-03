# Candidate K — SessionBench + Resident-Slot SLA

## 1. Thesis

Candidate K is a benchmark-first infrastructure company that sells conversational and agentic inference in the buyer's own unit of account: the **resident session-slot**. It (1) publishes **SessionBench**, an open, reproducible multi-turn session-serving benchmark that reports *sessions-per-GPU at a p95-TTFT SLO*, with collapse detection over a 40-minute full-load window — a number no engine, provider, or voice platform in the research files publishes (05-research-voice.md, Unknowns); (2) ships the residency runtime that wins it — a session-granular pinned-RAM KV connector plus a client-to-server idle-hint pinning protocol, the only mechanism that moved the concurrency knee in either repo; and (3) sells that capacity as **warm session-slots with a stated retention window, billed per slot-hour by tier**, into voice and agent platforms that already sell concurrency per line (Vapi $10/line/month; Retell $8/concurrency/month) and hosted agents per running hour (Anthropic Managed Agents $0.08/session-hour) — in the Wafer motion (engagements + dedicated endpoints; Wafer: 8-person team page, $40M Series A on 2026-09-01, customer logos including Vapi, Tavus, Inworld, DigitalOcean) rather than a self-serve token API. The sentence a customer would repeat: "They measured our cliff, moved it, and now we buy warm seats instead of best-effort cache hits."

Why this is not A–J. A and B are providers selling to end builders; K sells upstream to the platforms and providers themselves and is workload-agnostic (voice and agent trace profiles are swappable). C sells a scheduler as software; K's product is the *standard plus the contract*, and it is engine-neutral: SessionBench scores LMCache, vLLM's OffloadingConnector, SGLang HiCache and Dynamo KVBM alongside K's runtime, and if an OSS stack wins, K still has certification, slot pricing and the policy layer to sell. F's open-source artifact is a KV layer; K's is the harness. The honest overlap is C's "optimization to providers" leg; the difference is that K's demand engine is public measurement (Wafer's KernelArena and benchmark-post motion; Morph's throughput-claim motion) and its commercial primitive is time-priced residency, which no provider in 05-research-agents.md or 05-research-caching.md sells.

## 2. Customer and workload

**Buyer 1 — voice platforms that self-host an open LLM.** LiveKit serves Gemma 4 31B on SGLang with speculative decoding at 192 ms TTFT, priced $0.40/$1.20 per 1M tokens, and states it "reserves more headroom than a throughput-maximized deployment would." Inworld serves Gemma 4 26B and models the LLM at $0.000198/min, under 5% of a voice minute. ElevenLabs hosts Qwen3.6-35B-A3B and Qwen3.5-397B-A17B; Deepgram offers nemotron-3-nano-30B-A3B; Bland bundles proprietary models at $0.11–0.14/min. Every orchestrator fetched except Bland accepts a BYO OpenAI-compatible LLM endpoint. The unit they sell is concurrency: Vapi 10 lines then $10/line/month; Retell 20 then $8/concurrency/month; ElevenLabs 4–40 by plan with 2x burst pricing; Deepgram 45/60/100+; LiveKit 5/20/600 agent sessions. Workload shape (voice repo, synthetic): system prompt N(800,150) tokens, ~2.5k tokens per session (~140 MB KV at Qwen2.5-7B's 57,344 B/token), 6–30 turns, 300 ms mean inter-arrival, 95.2% compute-idle with idle windows p50 24.0 s — synthetic distributions never validated on real calls.

**Buyer 2 — agent platforms and open-model providers.** TraceLab (~4,300 Claude Code/Codex sessions): median prefix 126,180 / 115,584 tokens per step, ~857/886 new input tokens, 252/184 output; human gap median 1.4 min, p90 20.6 min; prefix-cache hits 95.7% overall but 84.4% on user-initiated steps. Codex traces: inter-turn 5.2 s median, 81.4 s P99, 131:1 input:output. They pay per token with best-effort caching (Fireworks: cache "only works within 1 replica," hits "usually... several minutes," 50% discount; OpenAI and xAI quoted in §3). Hosted agent runtimes already bill time: Managed Agents $0.08/session-hour while running, idle free; AgentCore 15-minute default idle timeout.

**Buyer 3 (engagement channel) — inference providers and clouds** wanting their session knee measured and moved (Wafer's Neon Health case: TTFT 800 ms to ~550 ms).

## 3. The pain, in the customer's words

- "It doesn't go off a cliff when you increase the requests per minute." — Neon Health testimonial on wafer.ai (the cliff is what buyers fear; the voice repo measured it: baseline prefix-cache hit rate 92% to 1.6–1.9% between N=200 and N=300).
- "prompt_cache_key... do not pin requests to a machine or guarantee a cache read hit" — OpenAI docs; "Cache entries can be evicted at any time due to server load or restarts" — xAI docs.
- "cached_input_tokens = 0" past ~150K tokens on Azure, 71% of input re-billed — openai/codex issue #25604; 17.1% higher spend after a TTL default change — anthropics/claude-code issue #46829.
- "orchestrators know session structure and tool-gap duration that request-local LRU cannot observe" — SGLang RFC #27574 (open, unimplemented).
- "Memory your engine doesn't know about will kill it" and "a 10-minute load test would have called the config healthy" — voice-inference docs/FINDINGS.md §4 and RESULTS.md.
- Building this in-house "takes 20 engineers and three or four months" — Tensormesh CEO (TechCrunch).

## 4. Value proposition and the proof-of-value benchmark

**Metric.** N* = the largest concurrent session count per GPU at which full-load-window (minutes 5–40) p95 TTFT ≤ 300 ms, reported as median and spread over 3 seeds, plus time-to-collapse; and cost per resident slot-hour = GPU $/hr ÷ N*.

**Setup.** The voice-inference harness as it ran: L40S 46 GB, Qwen2.5-7B fp16, vLLM, sessions = 5N+20, 60 s warmup discarded, TTFT measured client-side from STT-final to first chunk, time_scale 1.0 (evidence_grade). Extended by K to H100-80GB, one 26–35B model, and an agent-trace profile.

**Baselines, by name, and what is measured today (all seed 1, L40S).** Vanilla vLLM with prefix caching: N*=200 (p95 93.1 ms), collapse between 200 and 300. LMCache 0.3.2 default as a config change: N*=200, N=300 p95 25,997.7 ms; layerwise+save_decode 6,025.1 ms. VoiceKVConnector (session-granular contiguous pinned-RAM): N=300 p95 316.8 ms whole-run / 336.8 ms full-load — stable but fails the gate. Arm D (connector + idle-hint pinning, 10 GB budget): N=300 p95 272.2 ms whole-run / 291.9 ms full-load, N=400 p95 34,048 ms — so N*=300, a 1.5x capacity ratio at one seed with 3 of 7 full-load buckets above 300 ms. Unmeasured baselines K must add: vLLM OffloadingConnector (lru/arc), LMCache 0.5.4, SGLang HiCache L2, Dynamo KVBM G2; hosted endpoints (Fireworks, Wafer, LiveKit Inference, Tensormesh) through the same client harness, which yields a knee-at-RPM curve, not sessions/GPU.

**Target.** N* ≥ 1.4x the best OSS baseline, median of 3 seeds, on both a 46 GB and an 80 GB GPU, with full-load p95 ≤ 300 ms in ≥ 6 of 7 five-minute buckets. This is deliberately below the voice PRD's 2x "proceed" line, which the measured 1.5x did not clear.

**Why a skeptical buyer believes it.** Raw events.jsonl for 287,015 measured turns across 19 runs are archived; gates were pre-registered; the harness carries guards that each caught a real false result (512-thread client pool, hard-fail on zero turns, time_scale 1.0 only, per-point archive); and SessionBench runs on the buyer's own traces and GPU. The score that sells is the buyer's cliff, reproduced, then moved.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| SessionBench harness | Custom (from voice-inference `vkv/`) | Trace replay with turn-taking/barge-in, N-slot loadgen, JSONL events, manifest, analyzer with full-load-window p95, knee search, multi-seed aggregation |
| Trace profiles | Custom | Synthetic voice (existing), real-transcript converter (new), agent profile shaped on TraceLab stats (new), shared-prompt confound |
| Engine | OSS vLLM / SGLang, unforked | Prefill, decode, paged KV, prefix cache |
| Residency connector | Custom (VoiceKVConnector, 574 lines, ported to vLLM 0.28) | Save prompt KV per session as one contiguous D2H into a pinned slab; restore as one H2D + scatter; content-addressed prefix chain |
| Retain protocol | Custom | Client sends `kv_transfer_params.idle_ms` (later `retain_for_ms`); `request_finished` delays block free under a per-GPU budget; maps onto SGLang RFC #27574 Pin-with-TTL if it lands |
| Admission / slot budget | Custom | Hard cap on resident bytes; all speculation capped (the un-capped bypass OOM'd the engine at 345 concurrent requests) |
| Tier break-even table | Custom (dexa `evals/stateful_cost_model.py`, `dexa_platform/sessions/tiering.py`) | Which tier can honor which retention window at what cost |
| SLA meter and billing | Custom (dexa `dexa_platform/control/`) | Resident slot-hours by tier, hit/miss accounting, hashed keys, daily rollups |
| Portable KV format | Custom (dexa `src/dexa/session/`) | Cross-instance resume (bit-identical at TP=1), fp8 storage at 0.5x bytes |
| Gateway / affinity | OSS llm-d session affinity or dexa `edge/` | Session-to-replica stickiness |

Differentiation lives in the **measurement standard and the residency-policy layer** (scheduler-side connector hooks + admission control + time-priced retention) — not in kernels, not in the model. Nothing in vLLM/SGLang is replaced; LMCache's LRU-only eviction is replaced by hint-driven pinning; the PRD's never-built "free at turn end" is added as the inverse of pinning.

```
 platform orchestrator ──traces──▶ SessionBench (replay/loadgen/metrics/knee) ──score──▶ cliff report
   │ retain_for_ms per request                                                          
   ▼                                                                                    
 vLLM / SGLang (unforked) ── residency connector (pinned-RAM slab) ── retain/pin + slot budget
   │                                  └── tier break-even policy (HBM / RAM / NVMe)
   ▼
 SLA meter (resident slot-hours by tier) ──▶ billing / dedicated endpoint quote
```

## 6. Evidence

### Proven
- Baseline vLLM knee on L40S/Qwen2.5-7B: N=200 p95 93.1 ms; N=300 congestion collapse; prefix-hit 92% → 1.6–1.9%. (02-evidence-ledger-voice.md, vi-armA-baseline-knee; runs/modal/sweep_a)
- LMCache 0.3.2 as a config change did not move the knee: N=200 170.6 ms; N=300 25,997.7 ms default, 6,025.1 ms layerwise. (vi-armB-lmcache-reactive)
- LMCache CPU restore beats cold prefill single-request in vLLM 0.24: 10.0x at 4k (301→30 ms), 17.1x at 16k (1,320→77 ms), 22.0 GB/s. (01-evidence-ledger-dexa.md, lmcache-offload-restore)
- vLLM prefix-cache hit vs cold: ~12x at 4k, 25–34x at 16k; ≥32k crashed the engine. (vllm-warmstart-prefix-cache)
- Cross-instance resume bit-identical at TP=1; fp8/int8 KV at 0.5x bytes with 100% greedy agreement over 48 tokens. (cross-instance-resume; kv-interchange-formats)
- Connector conforms to vLLM 0.24 KVConnectorBase_V1, 10/10 hooks. (connector-conformance)
- Harness reality: 287,015 turns over 19 runs; 27 tests pass once pytest-asyncio is installed. (03-build-inventory.md)

### Bounded / contradicted
- Arm D 1.5x (N*=300 vs 200): single seed, single GPU, 27.8 ms whole-run margin / 8.1 ms full-load margin, 3 of 7 buckets >300 ms; N=400 fails (34,048 ms). The repo's README thresholds put 1.5x in the "kill" band; RESULTS.md re-scores it "above kill gate, below proceed." (vi-armD-predictive-pinning; vi-armD-n400-failure; 04-conflicts.md §8)
- VoiceKVConnector vs LMCache 19–82x at the N=300 tail is a stable-vs-collapsed comparison, not a per-transfer number; the "7 ms restore" is a docstring estimate, no transfer timings archived. (vi-contiguous-vs-chunked-transfer)
- Restore-vs-re-prefill is **contradicted** across the dexa repo: 10–34x single-request pinned-RAM vs 0.37x single-request and P99 8,930 vs 5,596 ms under 24-way concurrency for the disk-backed Dexa connector. Reconciliation (derived, 04-conflicts.md §1): restore wins iff tier bandwidth × vLLM prefill time > KV bytes; pinned RAM at 10–20 GB/s crosses over below 4k, disk at <1 GB/s only near 64k; RAM-tier restore at 16k–64k under concurrency is unmeasured anywhere.
- 95.2% idle (Phase 0) is synthetic, computed at 131,072 B/token and 60 tok/s versus the served 57,344 B/token and ~45 tok/s. (vi-phase0-duty-cycle; 04-conflicts.md §13)
- Cost-model break-evens (64k: HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr; "2–6x cheaper") are modeled at an assumed $1.80/GPU-hr with HF prefill times; not measured. (residency-cost-model)
- Proactive free-at-turn-end (bounded working set, ~4 GB projected at N=300) was described but never built or run. (04-conflicts.md §9)
- Audio signals contributed nothing measurable; arm D uses response length + turn statistics only. (vi-audio-signal-contribution)
- Server-side hit rates (35→45%), "8,000+ restores" and queue depths for B/C/D are unarchived. (02-evidence-ledger-voice.md discrepancy 7)

### Unproven

| Claim | Experiment that proves or kills it | Est. cost |
|---|---|---|
| 1.5x replicates across seeds | Arms A and D, seeds 1–3, N ∈ {200,250,300,350} on L40S | ~24 GPU-h, 4 days |
| Knee/ratio survives an 80 GB GPU | Arm A and D at N=300/400/500 on H100-80GB (derived: ~60 GB pool fits the ~42 GB working set, cliff ~N 400–450, not run) | ~10 GPU-h, 2 days |
| Real voice traffic is idle-dominated | Converter from partner transcripts to SessionTrace JSONL; rerun Phase 0 at 57,344 B/token and 45 tok/s; one arm A/D point | ~5 GPU-h, partner-gated, 2 weeks |
| Agent profile (100k+ prefix, 1.4 min gaps) benefits in-engine | vLLM 0.28 / SGLang 0.5.18, 32k–128k contexts, concurrency 8–128, RAM-tier restore vs re-prefill (the unrun crossover surface of 04-conflicts.md §1) | ~30 GPU-h, 1 week |
| 2026 OSS stacks already hold N=300 | vLLM OffloadingConnector (lru/arc), LMCache 0.5.4, SGLang HiCache, Dynamo KVBM on the same harness | ~30 GPU-h, 1–2 weeks |
| A retention window can be honored | Pin budget vs adversarial load; server-side hit rate scraped into events.jsonl; % of retained sessions warm at return | ~10 GPU-h, 3 days |
| Free-at-turn-end bounds the working set | Inverse-pin implementation; HBM pool occupancy over the run; N=300/400 | ~8 GPU-h, 3 days |
| fp8 KV compounds with pinning | Same rig, fp8 KV on, arms A/D | ~5 GPU-h, 1 day |
| Restore path is correct | Archive `evals/modal_kv_correctness.py` output | ~1 GPU-h |
| Platforms will pay per slot-hour | 10 discovery calls; 2 design partners run SessionBench on their stack | 0 GPU-h, 3 weeks |

## 7. MVP and 6-week build plan

**Ships first:** SessionBench v0.1 (public leaderboard + methodology) and one design-partner cliff report; the residency runtime as a vLLM 0.28 connector with the retain protocol; a slot-hour meter. Target GPU costs above total ~120 GPU-h.

- **Week 1 — harness generalization.** Reuse `/home/user/voice-inference/vkv/{traces,orchestrator,loadgen,metrics,run.py,sweep.py}` and `evals/modal_arm_a.py`, `scripts/run_sweep_resilient.sh`. New: text prompts + tokenizer + chat endpoint, aiohttp mid-stream abort, server-metrics scraping into events.jsonl, connector/flags/git-sha in manifests, multi-seed aggregation, full-load-window p95 as the reported number, missing-TTFT turns counted as failures. Add pytest-asyncio to requirements.
- **Week 2 — SessionBench v0 runs.** Arms A / LMCache 0.5.4 / OffloadingConnector / VoiceKV / D on L40S and H100, 3 seeds, N sweep; archive server logs. Publish the 12-guard methodology from docs/FINDINGS.md §4.
- **Week 3 — runtime port.** Port `vkv/backends/voice_kv_connector.py` (pin logic ~40 lines around `request_finished`, lines 173–183 and 348–379; 6-line client hint in `vkv/run.py`) to vLLM 0.28's KVConnector V1; implement free-at-turn-end; archive the correctness suite; adopt dexa's async-load pattern from `src/dexa/engine/vllm_connector.py`.
- **Week 4 — slot SLA layer.** `retain_for_ms` on `/v1/sessions` (API shape from `/home/user/dexa/dexa_platform/sessions/service.py`; tier logic from `sessions/tiering.py` and `edge/src/tiering.ts`); resident slot-hour meter on `dexa_platform/control/metering.py`; streaming passthrough (gateway currently forces `stream=False`).
- **Week 5 — design partner #1.** A voice platform self-hosting on vLLM/SGLang: transcript converter, SessionBench on their stack, cliff report, arm-D-style fix.
- **Week 6 — publish and price.** Leaderboard v0.1, partner report (with permission), dedicated session endpoints quoted per slot-hour; evaluate kill gates.

New code not in either repo: text/agent trace generator, metrics scraper, leaderboard site, SLA meter, vLLM 0.28 port, SGLang adapter, streaming, real-trace converter, tenant-namespaced KV keys.

## 8. Pricing model

Sell **resident slot-hours by tier** (HBM-pinned, RAM-warm, NVMe-warm) plus tokens at a pass-through rate, and Wafer-style engagements for measuring and moving a customer's cliff. The architecture makes the price expressible because the pin budget is a hard per-GPU count (10 GB ≈ 70 sessions of ~140 MB in the ledger's arithmetic) and N* is measured (300 per L40S at seed 1), so slot cost = GPU $/hr ÷ N*. Illustration only: at the cost model's assumed $1.80/GPU-hr (an A100-class assumption, not an L40S quote — L40S pricing is not in the research files), 300 slots gives $0.006 per slot-hour, against Managed Agents' $0.08/session-hour, Gemini's $0.50–$4.50 per 1M cached tokens per hour, and Anthropic's 2x write multiplier for a 1-hour TTL. The tier break-evens (4k: HBM ~1 min, RAM ~6 min, NVMe ~33 min; 64k: ~2 min, ~11 min, ~4.9 hr, all modeled) define which tier can honor which retention window at what cost. A per-token provider cannot express this: token billing has no time dimension, and every cache discount in the research files is best-effort (Fireworks, OpenAI, xAI, DeepSeek "within a few hours to a few days"). Tensormesh's cached-input-at-$0 and its post-v1 "30% of estimated savings" formula are the closest precedents; neither states a retention guarantee (guarantee-vs-target is a founder decision, §11).

## 9. Competitive facts

| Who | Adjacent shipping | Not shipped (per research files) | Source |
|---|---|---|---|
| Wafer | Optimization agents + serverless/dedicated inference; voice customers; KernelArena benchmark; TTFT 800→550 ms case | No session/KV-retention product; no sessions-per-GPU metric | 05-research-wafer.md |
| Tensormesh / LMCache | Cached input $0; persistent KV storage; on-prem Operator "coming soon"; LRU-family eviction | No TTL/idle-aware eviction; no retention guarantee; no named customers | 05-research-kvcache.md |
| vLLM | OffloadingConnector (lru/arc, CPU/FS/S3/P2P); KVConnector V1 "experimental" | No pin/retain hint API | 05-research-kvcache.md |
| SGLang | HiCache L1/L2/L3; RFC #27574 Pin-with-TTL (open); #24656 agent hints (inactive) | Neither RFC implemented | 05-research-kvcache.md |
| Mooncake | Store with 10 s leases, soft-pin ≤24 h; draft PR #2835 TTL retention leases | Draft only | 05-research-kvcache.md |
| NVIDIA Dynamo | KVBM G1–G4, presence_lfu filters, "cache_control-style TTL retention semantics" | No session benchmark; no per-slot pricing | 05-research-kvcache.md, 05-research-agents.md |
| llm-d | Session affinity + flow control (production v0.8), P2P KV pull, precise prefix routing | No retention SLA | 05-research-kvcache.md |
| Fireworks / Baseten | Session-affinity headers; KV-aware routing; CPU offload | Cache "not guaranteed"; per-replica | 05-research-agents.md |
| Anthropic / OpenAI / Gemini | 5m/1h TTL at 1.25x/2x; 30m/24h retention; storage per token-hour; Managed Agents $0.08/h | No guaranteed pinned KV SKU | 05-research-caching.md |
| Morph | Sticky KV placement via x-session-id; 50% standby tier | No retention guarantee | 05-research-morph.md |
| LiveKit / Inworld / ElevenLabs | Self-hosted open LLMs; concurrency-priced plans | No sessions-per-GPU figure; LiveKit over-provisions headroom | 05-research-voice.md |
| Artificial Analysis / SemiAnalysis | Caching table; tok/s-per-chip at tok/s-per-user | No multi-turn session-capacity benchmark | 05-research-kvcache.md, 05-research-wafer.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| 1.5x is seed noise | Median D/A capacity ratio over 3 seeds, L40S | < 1.2x (sell benchmark + engagements only) | ≥ 1.4x |
| 80 GB GPU dissolves the wall | H100 arm A knee vs L40S; D gain on H100 | A knee ≥ 2x L40S and D < 1.2x on H100 | D ≥ 1.3x on H100 |
| 2026 OSS already wins | Best of OffloadingConnector / LMCache 0.5.4 / HiCache / KVBM | Any holds N=300 ≤ 300 ms p95 out of the box (runtime leg becomes policy-only) | None do |
| Real calls are not idle | Phase 0 on partner transcripts (57,344 B/token, 45 tok/s) | Idle fraction < 40% | > 60% |
| Retention cannot be honored | % of retained sessions warm at return under a pin budget, adversarial load | < 95% | ≥ 99% |
| Agent leg has no in-engine win | RAM-tier restore vs re-prefill at 64k, 16-way concurrency | < 2x p99 TTFT improvement | ≥ 3x |
| Nobody buys slots | Design partners running SessionBench within 6 weeks | < 2 of 10 | ≥ 2, one paying |
| Connector API churn | Port effort vLLM 0.9.2 → 0.28 | > 2 engineer-weeks per major version | ≤ 1 week |

## 11. Founder decisions

- **Neutral benchmark vs. owning the winning engine.** Options: strictly neutral with a separate paid runtime; benchmark as marketing for our runtime; benchmark only. Evidence: whether OSS stacks win the Week-2 run.
- **Voice-first vs. agent-first trace profile.** Voice has the only measured knee (2.5k-token sessions, L40S); agents have public telemetry but zero in-engine evidence above 16k. Evidence: the two Week-2 profiles.
- **Upstream (platforms/providers) vs. downstream (end builders, i.e. B).** Evidence: discovery calls; Inworld models the LLM at <5% of a voice minute, Retell's LLM line is 4–65% — the spend a slot price competes for varies by platform; no market-size judgment is made here.
- **Engagements + dedicated endpoints (Wafer) vs. software/BYOC (Tensormesh Operator) vs. OSS harness + paid certification.** Evidence: whether partners can run the harness themselves.
- **GPU class.** L40S is where the wall exists; 80 GB moves it (derived, unrun). Evidence: the H100 gate.
- **Model class.** Everything measured is a 7B fp16 dense model; platforms run 26–35B and A3B MoE. Evidence: one 31B/A3B rerun.
- **Engine allegiance.** vLLM connector (experimental API) vs. SGLang RFC #27574 path vs. Dynamo KVBM retention semantics. Evidence: port cost and RFC status.
- **Guarantee vs. target.** Sell retention with SLA credits or as best-effort-with-metrics. Evidence: the honor-rate gate.
- **Whether Tensormesh's, llm-d's, Wafer's or Dynamo's adjacency matters.** Facts in §9; judgment yours.
- **Open-source the harness or not.** Evidence: whether platforms require neutrality to run it.

## 12. Combinations

- **With B (voice provider):** K's slot-hour is B's price list; B is K run downstream.
- **With C (residency scheduler):** C's scheduler is K's enforcement engine; K supplies C's proof-of-value and its billing unit.
- **With A (long-idle agents):** A's product is a retention window; K's tier break-even table and SLA meter are A's pricing engine once the agent-profile gate passes.
- **With F (open-core KV layer):** F's connector exposes `retain_for_ms`; K's benchmark is F's adoption channel.
- **With J (preemptible GPUs):** the NVMe/portable-state tier that K prices is J's evacuation path (cross-instance resume proven at TP=1).
- **With I (Morph playbook):** a latency-tuned specialized voice or agent model can sit on K's runtime; K's benchmark is I's proof.