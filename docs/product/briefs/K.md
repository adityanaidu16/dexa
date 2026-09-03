# Candidate K — SessionBench + Resident-Slot SLA (verified revision)

## 1. Thesis

K is a benchmark-first infrastructure company that sells conversational and agentic inference in the buyer's own unit: the **resident session-slot**. Three legs, ordered by evidence strength. (1) **SessionBench**: an open multi-turn session-serving benchmark reporting *sessions-per-GPU at a p95-TTFT SLO* with collapse detection over a full-load window (minutes 5–40). No engine, provider or voice platform in the research files publishes that metric (05-research-voice.md, Unknowns); adjacent public agent-trace benchmarks exist (vLLM/Mooncake on Codex traces, llm-d on 219 Claude Code sessions, VAST/Backend.AI at ~140K tokens) but report throughput, TTFT and hit rate, not capacity at an SLO. (2) A **residency runtime** — a session-granular pinned-RAM KV connector plus a client-to-server idle-hint pinning protocol — the only mechanism that moved the concurrency knee in either repo (1.5x, one seed, one 46 GB GPU). (3) **Warm session-slots with a stated retention window, billed per slot-hour by tier**, sold upstream to platforms that already price concurrency (Vapi $10/line/month; Retell $8/concurrency/month) and to hosted agent runtimes that bill time (Anthropic Managed Agents $0.08/session-hour while running), in the Wafer motion — engagements plus dedicated endpoints (Wafer: 8 people on its team page, $40M Series A on 2026-09-01, an *unlabeled* homepage logo carousel showing Vapi, Tavus, Inworld, DigitalOcean).

Honest status: leg 1 exists and produced 287,015 archived in-engine turns; leg 2 is one seed at 1.5x with an 8.1 ms full-load margin and, as measured, fails the pass criterion in §4; leg 3 is unbuilt. Time-priced retention is not novel: DeepInfra sells 5-minute/1-hour prompt-cache retention at 1.25x/2x write multipliers (no honor guarantee stated); Morph documents per-request TTL control; Mooncake soft-pins to 24 h. K's difference is the unit (a capacity slot at an SLO, on the buyer's traces) and the public measurement.

Why not A–J: A and B sell to end builders; K sells to platforms and providers. C sells a scheduler; K sells the standard plus the contract, engine-neutral — SessionBench scores LMCache, OffloadingConnector, HiCache and KVBM beside K's runtime; if OSS wins, K keeps certification, slot pricing and the policy layer. F's open artifact is a KV layer; K's is the harness.

## 2. Customer and workload

**Buyer 1 — voice platforms self-hosting an open LLM.** LiveKit serves Gemma 4 31B on SGLang with speculative decoding at 192 ms TTFT, $0.40/$1.20 per 1M tokens, and "reserves more headroom than a throughput-maximized deployment would." Inworld serves Gemma 4 26B and models the LLM at $0.000198/min, under 5% of a voice minute. ElevenLabs hosts Qwen3.6-35B-A3B and Qwen3.5-397B-A17B; Deepgram offers nemotron-3-nano-30B-A3B; Bland bundles proprietary models at $0.11–0.14/min. Every orchestrator fetched except Bland accepts a BYO LLM (OpenAI-compatible SSE; WebSocket for Retell). They sell concurrency: Vapi $10/line/month beyond 10; Retell $8/concurrency beyond 20; ElevenLabs 4–40 with 2x burst; Deepgram 45/60/100+; LiveKit 5/20/600 sessions. Measured workload (voice repo, synthetic only): system prompt N(800,150) tokens, ~2.5k tokens per session (~140 MB KV at Qwen2.5-7B's 57,344 B/token), 6–30 turns, 300 ms mean inter-arrival. The 95.2% compute-idle figure (idle windows p50 24.0 s) came from a different trace file (500 ms inter-arrival, 20 customers) at 131,072 B/token and 60 tok/s versus the served 57,344 B/token and ~45 tok/s; never validated on real calls; speech-to-speech untested.

**Buyer 2 — agent platforms and open-model providers.** TraceLab (~4,300 Claude Code/Codex sessions): median prefix 126,180 / 115,584 tokens per step, ~857/886 new input tokens, 252/184 output; human gap median 1.4 min, p90 20.6 min; prefix-cache hits 95.7% overall, 84.4% on user-initiated steps. Codex traces: inter-turn 5.2 s median, 81.4 s P99, 131:1 input:output. They buy tokens with best-effort caching (Fireworks: cache "only works within 1 replica," hits "usually... several minutes," 50% discount). Managed Agents bills $0.08/session-hour running, idle free; AgentCore idles out at 15 minutes.

**Buyer 3 (engagement channel) — inference providers and clouds** wanting their session knee measured and moved (Wafer's Neon Health case: TTFT 800 → ~550 ms).

## 3. The pain, in the customer's words

- "It doesn't go off a cliff when you increase the requests per minute." — Neon Health on wafer.ai. The voice repo measured the cliff: prefix-cache hit rate ~92% at N≤200, 1.6–1.9% at N=300.
- "prompt_cache_key... do not pin requests to a machine or guarantee a cache read hit" — OpenAI; "Cache entries can be evicted at any time" — xAI.
- "cached_input_tokens = 0" past ~150K tokens on Azure, 71% of input re-billed — openai/codex #25604; 17.1% higher spend after a TTL default change — anthropics/claude-code #46829 (community-measured, medium confidence).
- "orchestrators know session structure and tool-gap duration that request-local LRU cannot observe" — SGLang RFC #27574 (open; POC is a draft Mooncake PR).
- "Memory your engine doesn't know about will kill it"; "a 10-minute load test would have called the config healthy" — voice-inference FINDINGS.md §4, RESULTS.md.
- In-house "takes 20 engineers and three or four months" — Tensormesh CEO (TechCrunch).

## 4. Value proposition and the proof-of-value benchmark

**Metric.** N* = largest concurrent session count per GPU at which full-load-window p95 TTFT ≤ SLO, median and spread over 3 seeds, plus time-to-collapse and fraction of turns without a TTFT (counted as failures), reported as a curve over SLO ∈ {200, 300, 400, 500 ms} because the measured advantage is threshold-sensitive. Slot-hour cost = GPU $/hr ÷ N*.

**Setup as it ran.** Modal L40S 46 GB, Qwen2.5-7B-Instruct fp16, vLLM 0.9.2, --max-num-seqs 256 (unrecorded for the arm A sweep), sessions = 5N+20, 5 distinct system prompts, 60 s warmup discarded, TTFT client-side from STT-final to first chunk, time_scale 1.0, seed 1 only, manifests with git_sha "unknown" and no connector field. K extends to H100-80GB, one 26–35B model, an agent profile.

**Baselines measured today (seed 1, L40S).** Vanilla vLLM + prefix caching: N*=200 (p95 93.1 ms whole-run, 98.8 full-load); N=300 aborted after ~70 min with 82–111 running + 123–146 waiting — collapse inferred, no p95 exists. LMCache 0.3.2 as a config change (two configs, untuned): N*=200; N=300 p95 25,997.7 ms default, 6,025.1 layerwise+save_decode. VoiceKVConnector: N=300 p95 316.8 whole-run / 336.8 full-load, 6 of 7 buckets >300 ms — stable, fails the gate. Arm D (connector + idle-hint pinning, 10 GB budget): N=300 p95 272.2 / 291.9 full-load, N=400 34,048 ms; N*=300, 1.5x, 3 of 7 buckets >300 ms (311.8, 306.5, 303.9). The policy delta over the plain connector is 44.6 ms at p95, ~0 at p50; at a 350–400 ms SLO both hold N=300 and their ratio is 1.0x. Unmeasured baselines to add on the *same* engine version: OffloadingConnector (lru/arc), LMCache 0.5.4 tuned, SGLang HiCache L2, Dynamo KVBM G2. Hosted endpoints through the same client yield knee-at-RPM, not sessions/GPU — a separate track.

**Target.** N* ≥ 1.4x the best same-version OSS baseline, median of 3 seeds, on 46 GB and 80 GB GPUs, with full-load p95 ≤ 300 ms in ≥ 6 of 7 buckets. **As measured, no arm clears this**: arm D's 4-of-7 is the hypothesis tuning (16 GB pin budget, N=350) must convert. The 1.4x sits below the voice PRD's 2x proceed line, which the measured 1.5x did not clear and which the README scores as "kill."

**Why a skeptical buyer believes it.** Raw events.jsonl for 287,015 turns over 19 runs are archived; gates were pre-registered; harness guards each caught a real false result (512-thread client pool, hard-fail on zero turns, time_scale 1.0 only, per-point archive); SessionBench runs on the buyer's traces and GPU. The skeptic also sees: single seed, synthetic traces, unarchived server hit rates and correctness output, a year-old engine — fixed before v0.1 publishes.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| SessionBench harness | Custom (voice-inference `vkv/`) | Trace replay with turn-taking/barge-in, N-slot loadgen, JSONL events, manifest, analyzer; new: full-load p95, SLO curve, multi-seed, server-metrics scrape |
| Trace profiles | Custom | Synthetic voice (exists); transcript converter, agent profile on TraceLab stats, shared-prompt confound (new) |
| Engine | OSS vLLM / SGLang, unforked | Prefill, decode, paged KV, prefix cache |
| Residency connector | Custom (VoiceKVConnector, 574 lines, vLLM 0.9.2; **0.28 port is new**) | One contiguous D2H per prefill into a pinned slab; one H2D + scatter on restore; blake2b prefix chain. Today: TP=1, fp16/bf16, non-MLA, prompt-KV only, synchronous loads |
| Retain protocol | Custom | Client sends `kv_transfer_params.idle_ms` (→ `retain_for_ms`); `request_finished` delays block free under a per-GPU budget; maps onto SGLang RFC #27574 Pin-with-TTL if it lands |
| Admission / slot budget | Custom | Hard cap on resident bytes; all speculation capped (an un-capped bypass OOM'd the engine at 345 concurrent requests; pre-fix run unarchived) |
| Free-at-turn-end | Custom, **never built** | Inverse of pinning; the PRD's bounded-working-set mechanism (~4 GB projected, unvalidated) |
| Tier break-even table | Custom (dexa `evals/stateful_cost_model.py`, `sessions/tiering.py`) | Modeled, advisory; needs measured H100 inputs |
| SLA meter | Custom; reuse dexa `control/` key hashing + daily rollups | Slot-hours by tier and *observed* hit/miss are new (dexa infers warm = not first touch) |
| Portable KV format | Custom (dexa `src/dexa/session/`) | Cross-instance resume, bit-identical at TP=1 (OPT-125m, Qwen3-0.6B, A10G); bf16 blob |
| Gateway / affinity | OSS llm-d session affinity (production v0.8) | Session-to-replica stickiness; dexa `edge/` is a single-backend scaffold, never deployed |

Differentiation lives in the **measurement standard and the residency-policy layer** (connector hooks + admission + time-priced retention), not kernels or models. Nothing in vLLM/SGLang is replaced; LMCache's LRU-family eviction is joined by hint-driven pinning. Caveat: the larger measured effect (stable vs collapsed at N=300) came from the contiguous transfer path, not the policy.

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
- Baseline vLLM knee on L40S/Qwen2.5-7B: N=200 p95 93.1 ms; collapse between 200 and 300, prefix-hit ~92% → 1.6–1.9%. Single seed; N=300 p95 never measured; 5 prompts. (vi-armA-baseline-knee)
- LMCache 0.3.2 as a config change did not move the knee: N=200 170.6 ms; N=300 25,997.7 / 6,025.1 ms. Two configs, untuned. (vi-armB-lmcache-reactive)
- LMCache CPU restore beats cold prefill, single request, vLLM 0.24, A100, prefix caching OFF, LMCache version unpinned: 10.0x at 4k, 17.1x at 16k, 22.0 GB/s. (lmcache-offload-restore)
- vLLM in-GPU prefix-cache hit vs cold: ~12x at 4k, 25–34x at 16k; ≥32k crashed the engine. No transfer; the repo calls it table stakes. (vllm-warmstart-prefix-cache)
- Cross-instance resume bit-identical at TP=1 on OPT-125m and Qwen3-0.6B. fp8/int8 KV at 0.5x bytes with 100% greedy agreement — HF-level, Llama-3.1-8B, one passage, 48 tokens; not a blob-format feature. (cross-instance-resume; kv-interchange-formats)
- DexaConnector (not VoiceKV) conforms to vLLM 0.24.0 KVConnectorBase_V1, 10/10 hooks. (connector-conformance)
- Harness: 287,015 turns over 19 runs (recomputed); 27 tests pass with pytest-asyncio. (03-build-inventory.md)

### Bounded / contradicted
- Arm D 1.5x: single seed, single GPU, 27.8 ms whole-run / 8.1 ms full-load margin, 3 of 7 buckets >300 ms, N=400 fails; pin budget, hold multiplier and 25 s threshold are first guesses; the hint uses the trace's own median constants, so the predictor knows the generating distribution. README puts 1.5x in "kill"; RESULTS.md re-scores it "above kill gate, below proceed." (vi-armD-predictive-pinning; vi-armD-n400-failure; 04-conflicts.md §8)
- VoiceKV vs LMCache 19–82x at the N=300 tail is stable-vs-collapsed; "~7 ms restore" is a docstring estimate; transfer timings, server hit rates (35–52%, "35→45%"), "8,000+ restores" and queue depths are unarchived. (vi-contiguous-vs-chunked-transfer; discrepancy 7)
- Restore-vs-re-prefill is **contradicted** across dexa: 10–34x single-request pinned-RAM vs 0.37x single-request and P99 8,930 vs 5,596 ms at 24-way concurrency for the disk-backed DexaConnector. Derived reconciliation (04-conflicts.md §1): restore wins iff tier bandwidth × prefill time > KV bytes; pinned RAM at ~10–20 GB/s crosses below 4k, disk at <1 GB/s near 64k; RAM-tier restore at 16k–64k under concurrency is unmeasured anywhere.
- Code-reading finding (03-build-inventory.md): a store evicts the slab entry it *matched*, so sessions sharing a customer prompt evict each other's entries every turn; effect on hit rates unquantified.
- Cost-model break-evens (64k: HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr; "2–6x cheaper") are modeled at $1.80/GPU-hr with HF prefill times. (residency-cost-model)
- Free-at-turn-end never built; audio signals contributed nothing measurable. (04-conflicts.md §9; vi-audio-signal-contribution)
- Restore-correctness suite exists; no output archived. (vi-kv-restore-correctness)

### Unproven

| Claim | Experiment that proves or kills it | Est. cost (own estimate) |
|---|---|---|
| 1.5x replicates and clears 6/7 buckets | Arms A, B-voice, D; seeds 1–3; N ∈ {200,250,300,350}; pin 10/16 GB; L40S | ~40 GPU-h |
| Advantage survives an SLO curve | Same runs scored at 200/300/400/500 ms | 0 |
| Knee/ratio survives 80 GB | A and D at N=300/400/500 on H100 (derived: ~60 GB pool fits ~42 GB; cliff ~N 400–450) | ~15 GPU-h |
| 2026 OSS already holds N=300 | OffloadingConnector, LMCache 0.5.4 tuned, HiCache, KVBM on the same vLLM 0.28 / SGLang 0.5.18 as the port | ~40 GPU-h |
| Connector serves a customer-class model | One 26–35B (fp8 weights), H100, TP=1; MLA and TP>1 are new code | ~15 GPU-h + 2 eng-wk |
| Real voice traffic is idle-dominated | Partner-transcript converter; Phase 0 at 57,344 B/token, 45 tok/s; one A/D point | ~5 GPU-h, partner-gated |
| Predictor generalizes | Hint fitted on held-out sessions; barge-in ablation | ~6 GPU-h |
| Agent profile benefits in-engine | 32k–128k contexts, concurrency 8–128, RAM-tier restore vs re-prefill | ~30 GPU-h |
| Retention can be honored | % retained sessions warm at return under adversarial load; server hit rate in events.jsonl | ~10 GPU-h |
| Free-at-turn-end bounds the working set | Inverse-pin; HBM occupancy; N=300/400 | ~8 GPU-h |
| Restore correct; shared-prefix eviction fixed | Archive `evals/modal_kv_correctness.py`; per-session keying; --shared-prompt run | ~3 GPU-h |
| Platforms pay per slot-hour | 10 discovery calls; 2 design partners run SessionBench | 0 |

## 7. MVP and 6-week build plan

**Ships first:** SessionBench v0.1 (public methodology + leaderboard on synthetic traces, disclosed as such), the residency runtime as a vLLM 0.28 connector with the retain protocol, a slot-hour meter. GPU budget ~250 GPU-h (the sweep alone is ~120 runs of ~55 min; the draft's ~120 GPU-h undercounted), ≈ $600–900 at Modal's L40S $1.95/hr and H100 $3.95/hr.

- **Week 1 — port first, harness in parallel.** Port `/home/user/voice-inference/vkv/backends/voice_kv_connector.py` (pin config lines 179–183; `request_finished` 338–367; release sweep 369–379; `get_finished` 429–447; 6-line client hint `vkv/run.py:86–92`) to vLLM 0.28's KVConnector V1 (experimental; dexa needed live probes to find 0.24's 3-arg constructor). Reuse `vkv/{traces,orchestrator,loadgen,metrics,run.py,sweep.py}`, `evals/modal_arm_a.py`, `scripts/run_sweep_resilient.sh`. New: server-metrics scrape, provenance in manifests, multi-seed, SLO curve, missing-TTFT-as-failure, pytest-asyncio in requirements.
- **Week 2 — port lands or re-baseline.** If the port is not green, run every arm on 0.9.2 with LMCache 0.3.2 tuned, labeled "legacy engine"; never mix versions across arms. Archive the correctness suite; fix shared-prefix eviction.
- **Week 3 — SessionBench v0 runs.** A / LMCache 0.5.4 / OffloadingConnector / VoiceKV / D on L40S and H100, 3 seeds, N sweep, four SLOs; archive server logs.
- **Week 4 — slot SLA layer.** `retain_for_ms` on `/v1/sessions` (shape from `/home/user/dexa/dexa_platform/sessions/service.py`; tiers from `sessions/tiering.py`, `edge/src/tiering.ts`); meter on `dexa_platform/control/metering.py`; observed hit/miss; streaming (gateway forces `stream=False` at `gateway/app.py:127`); free-at-turn-end.
- **Week 5 — honor-rate and 26–35B model run**; agent-profile crossover slice; discovery calls throughout.
- **Week 6 — publish v0.1, evaluate gates.** A design-partner cliff report moves to Weeks 8–12: transcript conversion is partner-gated, and a partner signed after Week-1 discovery will not have traces converted by Week 5.

Realistic slip: 9–12 weeks if the port takes two weeks or the H100 gate forces a re-scope. New code in neither repo: agent trace generator, metrics scraper, leaderboard, SLA meter, 0.28 port, SGLang adapter, MLA/TP>1, streaming, transcript converter, tenant-namespaced keys.

## 8. Pricing model

Sell **resident slot-hours by tier** (HBM-pinned, RAM-warm, NVMe-warm) plus pass-through tokens, and Wafer-style engagements to measure and move a customer's cliff. The price is expressible because the pin budget is a hard per-GPU count (10 GB ≈ 70 sessions of ~140 MB) and N* is measured (300 per L40S, seed 1): slot cost = GPU $/hr ÷ N*. Illustration only: L40S list prices in 05-research-gpu-pricing.md run $0.79–0.99 (RunPod), $1.50 (Crusoe), $1.57 (DigitalOcean), $1.55–1.82 (Nebius), ~$1.95 (Modal), $2.25 (CoreWeave); at N*=300 that is $0.0026–0.0075 per slot-hour, against Managed Agents' $0.08/session-hour, Gemini's $0.50–4.50 per 1M cached tokens per hour, and Anthropic's 2x write multiplier for a 1-hour TTL. Tier break-evens (4k: HBM ~1 min, RAM ~6 min, NVMe ~33 min; 64k: ~2 min, ~11 min, ~4.9 hr; modeled at $1.80/hr A100 assumptions) define which tier can honor which window. Precedents: DeepInfra's 5m/1h retention at 1.25x/2x writes, 0.2x reads; Tensormesh's cached-input-at-$0 and post-v1 "30% of estimated savings"; neither states an honor guarantee (guarantee vs target is a founder decision). Token billing has no time dimension; every other cache discount in the research files is best-effort (Fireworks, OpenAI, xAI, DeepSeek).

## 9. Competitive facts

| Who | Adjacent shipping | Not shipped (per research files) | Source |
|---|---|---|---|
| Wafer | Optimization agents; serverless + dedicated; knee-at-RPS posts (GLM-5.2 "knee of ≤5s TTFT"); Neon Health 800→~550 ms | No session/KV-retention product; no sessions-per-GPU metric | 05-research-wafer.md |
| Tensormesh / LMCache | Cached input $0; "persistent KV cache storage", "session-scoped memory management"; Operator "coming soon"; LRU eviction | No TTL/idle-aware eviction documented; no retention guarantee; no named customers | 05-research-kvcache.md, 05-research-agents.md |
| DeepInfra | Explicit cache retention 5m/1h at 1.25x/2x write, 0.2x read; `prompt_cache_key` | No honor guarantee; two models; per-token unit | 05-research-providers.md |
| Morph | Sticky KV via x-session-id; per-request TTL control; 50% standby tier | No retention guarantee | 05-research-morph.md |
| vLLM | OffloadingConnector (lru/arc, CPU/FS/S3/P2P); KVConnector V1 "experimental" | No pin/retain hint API | 05-research-kvcache.md |
| SGLang | HiCache L1/L2/L3; RFC #27574 Pin-with-TTL (open); #24656 (inactive) | Neither RFC implemented | 05-research-kvcache.md |
| Mooncake | 10 s leases, soft-pin ≤24 h; draft PR #2835 TTL leases; 46x TTFT on Codex traces | Draft only; no capacity-at-SLO metric | 05-research-kvcache.md |
| NVIDIA Dynamo | KVBM G1–G4; presence_lfu; "cache_control-style TTL retention semantics" | No session benchmark; no per-slot pricing | 05-research-kvcache.md, 05-research-agents.md |
| llm-d | Session affinity + flow control (production v0.8); P2P KV; 219-session Claude Code study | No retention SLA | 05-research-kvcache.md |
| Fireworks / Baseten | Session-affinity headers; KV-aware routing; CPU offload | Cache "not guaranteed"; per-replica | 05-research-agents.md |
| Cerebras | 5-min guaranteed cache TTL (to 1 h) | No discount; gpt-oss-120b only | 05-research-providers.md |
| Anthropic / OpenAI / Gemini | 5m/1h TTL at 1.25x/2x; 30m/24h retention; storage per token-hour; Managed Agents $0.08/h | No guaranteed pinned-KV SKU | 05-research-caching.md |
| Continuum; "Stateful Inference" (arXiv) | TTL-pinned GPU KV, >8x JCT; persistent KV across agent turns, 2.1–4.2x | Papers, not products | 05-research-agents.md |
| LiveKit / Inworld / ElevenLabs | Self-hosted open LLMs; concurrency-priced plans | No sessions-per-GPU figure; LiveKit over-provisions headroom | 05-research-voice.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| 1.5x is seed noise / never clears 6-of-7 | Median D/A over 3 seeds, L40S, bucket rule | < 1.2x, or 6/7 never met | ≥ 1.4x and 6/7 |
| Advantage is an SLO artifact | D vs B-voice N* at 200/300/400/500 ms | D = B-voice at ≥3 of 4 SLOs (policy becomes a free feature) | D > B-voice at ≥3 |
| 80 GB dissolves the wall | H100 A knee vs L40S; D gain on H100 | A knee ≥ 2x L40S and D < 1.2x | D ≥ 1.3x |
| 2026 OSS already wins | Best of OffloadingConnector / LMCache 0.5.4 / HiCache / KVBM, same engine version | Any holds N=300 ≤ 300 ms (runtime leg becomes policy-only) | None do |
| Engine-version confound | Arms on mixed vLLM versions | Any mixed-version headline | One version for all arms |
| Customer-class model unserved | 26–35B on H100 with ported connector | Cannot run TP=1 fp8, or D < 1.2x | Runs, D ≥ 1.3x |
| Real calls are not idle | Phase 0 on partner transcripts (57,344 B/token, 45 tok/s) | Idle < 40% | > 60% |
| Predictor does not generalize | Held-out-session hint; ablation | D gain < 50% of in-sample | ≥ 80% |
| Retention cannot be honored | % retained sessions warm at return, adversarial load | < 95% | ≥ 99% |
| Restore wrong / shared-prefix thrash | Correctness archive; shared-prompt hit rate | Any missing-entry error; hit rate drops > 20 pts | Clean |
| Agent leg has no in-engine win | RAM-tier restore vs re-prefill at 64k, 16-way | < 2x p99 TTFT | ≥ 3x |
| Connector API churn | Port 0.9.2 → 0.28 | > 2 eng-weeks | ≤ 1 week |
| Nobody buys slots | Design partners running SessionBench | < 2 of 10 within 8 weeks | ≥ 2, one paying |

## 11. Founder decisions

- **The SLO.** 300 ms p95 TTFT is the PRD's number; Twilio budgets 375 ms target / 750 ms max for LLM TTFT; LiveKit reports 192 ms. The threshold decides whether the policy layer has a measured edge. Evidence: the SLO curve.
- **Pass criterion.** 6-of-7 buckets (current data fails) vs full-load p95 (passes by 8 ms) vs whole-run (27.8 ms). Evidence: seeds 2–3.
- **Neutral benchmark vs owning the winning engine.** Neutral harness + paid runtime; benchmark as marketing; benchmark only. Publishing OSS scores may hand the result to competitors.
- **Voice-first vs agent-first; trace source.** Voice has the only measured knee (2.5k-token sessions); agents have public telemetry (TraceLab, Mooncake traces) and zero in-engine evidence above 16k.
- **Upstream (platforms/providers) vs downstream (end builders, i.e. B).** Inworld models the LLM at <5% of a minute, Retell 4–65% — the spend a slot price competes for varies by platform. No market-size judgment made here.
- **Engagements + dedicated endpoints (Wafer) vs software/BYOC (Tensormesh Operator) vs OSS harness + paid certification.** Evidence: whether partners run the harness themselves.
- **Fork or not.** Scheduler-native prefetch is the untested strongest form of the idle thesis and needs the vLLM fork the PRD forbade.
- **GPU and model class.** L40S is where the wall exists; 80 GB moves it (derived). Everything measured is 7B fp16; platforms run 26–35B and A3B MoE, some MLA. Evidence: H100 and 31B gates.
- **Engine allegiance.** vLLM connector (experimental API) vs SGLang RFC #27574 vs Dynamo KVBM retention. Evidence: port cost, RFC status.
- **Pricing unit.** Slot-hour vs per-line/month (Vapi, Retell) vs cached-token retention (DeepInfra, Gemini) vs % of savings (Tensormesh). Evidence: discovery calls.
- **Guarantee vs target; single-tenant vs shared slots; tenant KV isolation.** Evidence: honor-rate gate.
- **Whether Tensormesh, DeepInfra, llm-d, Wafer or Dynamo adjacency matters.** Facts in §9; judgment yours.
- **Open-source the harness or not.** Evidence: whether platforms require neutrality.

## 12. Combinations

- **With B (voice provider):** K's slot-hour is B's price list; B is K run downstream.
- **With C (residency scheduler):** C is K's enforcement engine; K supplies C's proof and billing unit.
- **With A (long-idle agents):** A's product is a retention window; K's tier table and meter are A's pricing engine once the agent gate passes.
- **With F (open-core KV layer):** F's connector exposes `retain_for_ms`; K's benchmark is F's adoption channel.
- **With J (preemptible GPUs):** K's NVMe/portable-state tier is J's evacuation path (cross-instance resume proven at TP=1, small models).
- **With I (Morph playbook):** a latency-tuned specialized model on K's runtime; K's benchmark is I's proof.
