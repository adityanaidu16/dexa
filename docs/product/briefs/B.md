# Candidate B — Conversation-native inference for voice agents (verified revision)

## 1. Thesis

A voice-agent LLM inference company whose product metric is **concurrent voice sessions per GPU at p95 time-to-first-token (TTFT) <= 300 ms**, sold as a bring-your-own-cloud (BYOC) package or a hosted OpenAI-compatible endpoint (order: founder decision). Mechanism: session-semantic KV residency — each request carries the conversation's predicted idle window; a vLLM KVConnector plugin pins sessions predicted to return soon out of the engine's evictable pool and lets the rest evict into a contiguous pinned-CPU restore path, so eviction becomes a cheap load, not a re-prefill. No fork. Evidence in one line: the pinning arm is the only one of five configurations that moved the measured knee — once, seed 1, 46 GB L40S, 7B model, 1.5x, 8 ms full-load-window margin (02-evidence-ledger-voice.md, vi-armD-predictive-pinning, BOUNDED). The customer's sentence: "We know how many calls a GPU holds at p95 under 300 ms, and it doesn't fall off a cliff when we add lines."

## 2. Customer and workload

**Who buys.** (a) Voice platforms self-hosting an open-weight LLM: LiveKit (Gemma 4 31B on SGLang + speculative decoding, 192 ms TTFT, $0.40/$1.20 per 1M tokens), Inworld (Gemma 4 26B), ElevenLabs (Qwen3.6-35B-A3B, Qwen3.5-397B-A17B), Deepgram (Nemotron-3-nano-30B) (05-research-voice.md). (b) Their customers bringing their own LLM: every orchestrator fetched except Bland accepts a custom LLM over OpenAI-compatible SSE (Vapi, ElevenLabs) or WebSocket (Retell).

**What they run today.** Cascaded STT -> LLM -> TTS; the LLM is a 26-35B open model or a frontier API. The measured workload is far smaller than any documented production default: Qwen2.5-7B fp16, ~2.5k tokens per session, ~140 MB KV per session at 57,344 B/token, median reply 75 synthetic tokens (02-evidence-ledger-voice.md header). No vendor default at 7B was found.

**Turn and idle pattern.** Decode finishes far ahead of playback (measured ITL ~22 ms, ~45 tok/s, vs 3.25 tok/s speech). On synthetic traces the model is compute-idle 95.2% of wall clock and ~93% of idle windows exceed 8 s (5,133 of 5,491; vi-phase0-duty-cycle, BOUNDED: every timing distribution is assumed; computed at 60 tok/s and 131,072 B/token, neither the served model's). The 95% comes from 75-token replies taking ~23 s to play; shorter real replies shrink the window.

**Concurrency and payment.** Concurrency is sold as lines: Vapi $10/line/month beyond 10; Retell $8/concurrency/month beyond 20; LiveKit 5/20/50 inference concurrency by plan. The LLM line is paid per token (LiveKit), per minute by model (Retell $0.003-$0.16/min), at cost (Vapi, Inworld), or bundled (Bland $0.11-$0.14/min). Twilio budgets 375 ms target / 750 ms max for LLM TTFT in a ~1.1 s mouth-to-ear turn. The 300 ms SLA is the repo's PRD choice, not a customer number.

## 3. The pain, in the customer's words

- LiveKit, on its own Gemma 4 deployment, reserves "more headroom than a throughput-maximized deployment would", "accepting higher per-request cost" (LiveKit blog 2026-07-02); Inworld's throughput-oriented Gemma 4 31B config shows ~1.7 s p50 TTFT (05-research-voice.md).
- Neon Health (a voice agent, on Wafer): "it doesn't go off a cliff when you increase the requests per minute" (05-research-wafer.md). The repo measured a cliff: baseline vLLM serves 200 synthetic sessions at p95 93.1 ms and collapses between 200 and 300 as prefix-cache hit rate falls ~92% -> 1.6-1.9% (vi-armA-baseline-knee, PROVEN).
- Deepgram: the LLM "is usually the largest line." Inworld's model puts a self-hosted LLM under 5% of a minute (TTS 70-73%); smallest.ai ~30%; Retell's LLM line ~4-65% (derived). Capacity/TTFT pain for some stacks, cost for others.
- No vendor found advertises sessions-per-GPU for the LLM stage; the only per-GPU figures found are Kyutai STT (64/L40S, 400/H100) and Rime TTS (100+/machine). The search budget ran out before a dedicated search on this term (05-research-voice.md, Unknowns), so "nobody publishes this" is unverified.

## 4. Value proposition and the proof-of-value benchmark

**Metric.** Max concurrent sessions N at p95 TTFT <= 300 ms, client-side from STT-final to first streamed chunk, over the **full-load window** (minutes 5-40) as primary and whole-run as secondary, median and spread over 3 seeds, runs >= 40 min because collapse was depth-triggered at minute 15-20 (vi-harness-methodology-lessons). Turns without a TTFT count as failures (conflict 14).

**Setup.** The voice-inference replay harness: trace-driven sessions with turn-taking, barge-in (10%), spurious VAD (5%), full-history resubmission, 60 s warmup discard, archived events.jsonl per point (03-build-inventory.md). Traces: synthetic; no real-transcript converter exists yet.

**Baselines, by name.** (1) Vanilla vLLM with prefix caching (arm A); (2) vLLM's native OffloadingConnector (CPU tier, lru/arc; since 0.11.0; TTFT 2-22x on Llama-3.1-8B/H100 per the vLLM blog, 05-research-kvcache.md) — **unrun, and the first baseline a skeptical buyer will ask for**; (3) LMCache 0.5.4 layerwise — the measured collapse is LMCache 0.3.2 on vLLM 0.9.2, a 2025 stack; (4) SGLang + HiCache host tier (unrun; LiveKit's engine); (5) our connector without pinning (B-voice) and with pinning (D); (6) D with shuffled idle hints, separating predictor from pin. Hosted endpoints compare on TTFT only.

**Measured so far (seed 1, L40S 46,068 MiB, Qwen2.5-7B fp16, vLLM 0.9.2).** Max passing N: A 200, B-LMCache 200, B-voice 200, C 200, D 300 (sweep_*/results.json). At N=300: LMCache default p95 25,997.7 ms; B-voice 316.8 ms, stable; D 272.2 ms; D at N=400 34,048 ms. No point between 200-300 or 300-400 was run, so A's knee is in (200, 300), D's in [300, 400), and the true ratio lies roughly in [1.0x, 2.0x); "1.5x" is 300/200.

**Target number.** >= 2x the best of baselines (1)-(3) on the same GPU and model, median of 3 seeds, full-load-window p95. The repo disagrees with itself on thresholds: README's pre-registered table (for arm C vs A) has no 1.5-2x band ("< 2x kill · 2-3x conditional · > 3x proceed"); RESULTS.md re-scores D's 1.5x against the Phase-2 arm-B gate (>= 1.5x) as "above kill gate, below the 2x proceed line" (conflict 8; the README table is superseded). This brief pre-registers 2x; the ratio is a founder decision. Secondary target: no collapse, p95 under 1 s at 1.5x the baseline knee.

**Why a skeptical buyer would believe it.** Harness and raw logs exist (287,015 measured turns archived across 19 runs; to be published); baselines are the buyer's own stack; the plugin runs inside the buyer's vLLM, so they reproduce on their GPUs and transcripts. Not claimable yet: "2x", "across seeds", "on H100", "on 30B", "on real calls", "bit-correct restore" (suite exists, no output archived; vi-kv-restore-correctness, OPEN), "beats a current OffloadingConnector/LMCache".

## 5. Architecture

| Component | Custom / OSS | Role |
|---|---|---|
| Session KV connector | Custom (vLLM KVConnectorBase_V1 plugin, no fork) | Block-aligned prompt KV -> one pinned-CPU slab, one async D2H per request; blake2b prefix chain over 16-token blocks; one contiguous H2D + scatter on restore; delayed-free pin via request_finished/get_finished (voice-inference/vkv/backends/voice_kv_connector.py, 574 lines). Today: TP=1, fp16/bf16, non-MLA, prompt KV only, synchronous loads, pinned to vLLM 0.9.2 |
| Residency policy | Custom | idle_ms -> pin/hold/evict under a budget: pin if idle <= 25 s and the 10 GB budget allows, hold = idle x 1.3 + 2,000 ms (~70 sessions). Proactive free of predicted long-idle sessions: designed, not built (conflict 9) |
| Idle-window predictor | Custom | Today 6 lines: TTS playback remainder + the trace's own median gap (400 ms) and utterance (2,500 ms) + STT final (vkv/run.py _idle_hint) — it knows the generating distribution. Later: learned per-customer predictor |
| Session gateway | Custom (dexa_platform/gateway, 692 lines; edge/ Durable Object pattern, never deployed) | OpenAI SSE and Retell-style WebSocket; session-id -> replica affinity; injects kv_transfer_params.idle_ms; keys/metering (dexa_platform/control). Today: stream=False forced; no affinity map |
| Engine | OSS: vLLM (target 0.28.0); SGLang HiCache second | Unmodified scheduler, attention, kernels. KVConnector V1 is labeled experimental and has drifted (3-arg ctor at 0.24; batched CachedRequestData at 0.9.2) |
| Benchmark harness + correctness suite | Custom (vkv orchestrator/loadgen/metrics/sweep, ~2,000 lines; evals/modal_kv_correctness.py) | The proof-of-value instrument; bitwise slab round-trip + forced-evict probe, never archived |
| Server telemetry | New | Prefix-hit rate, queue depth, per-restore CUDA-event latency into events.jsonl (none archived) |

**Where the differentiation lives:** the residency decision and the gateway that carries conversational state into it — between engine and orchestrator, not in model, kernels or scheduler. Residency in LMCache, vLLM's OffloadingConnector and Mooncake is LRU-family; Mooncake has client-fixed soft-pin TTLs; SGLang RFC #27574 proposes Pin with bounded TTL; Morph exposes per-request cache TTL (05-research-kvcache.md; 05-research-morph.md). Whether a predicted idle window beats a client-fixed TTL is unmeasured.

```
 Orchestrator (Vapi / Retell / LiveKit agent / Pipecat)
      | SSE or WebSocket, session-id, STT-final text
      v
 +-----------------------------------------------------------+
 | Session gateway: affinity, tenant keys, metering,         |
 | idle_ms = f(reply length, turn stats, TTS rate)           |
 +----------------------+------------------------------------+
                        | /v1/completions + kv_transfer_params.idle_ms
                        v
 +-----------------------------------------------------------+
 | vLLM (unmodified)   scheduler --- KVConnector hooks        |
 |   HBM KV pool  <-pin/hold/free-  Session KV connector      |
 |        ^  one contiguous H2D        |  one async D2H       |
 |        +------- pinned-CPU slab (content-addressed) <------+
 +-----------------------------------------------------------+
   metrics: TTFT, hit rate, restores, queue depth -> events.jsonl
```

## 6. Evidence

### Proven

- Baseline vLLM cliff on L40S/7B: N=200 passes at p95 93.1 ms (18,324 turns); N=300 collapses (aborted after ~70 min; p95 never measured); prefix hit 92% -> 1.6-1.9% (vi-armA-baseline-knee). Caveats: single seed; 5 distinct system prompts; --max-num-seqs unrecorded; git_sha 'unknown'.
- CPU restore beats re-prefill single-request in a real vLLM stack: LMCache 10.0x at 4k (301 -> 30 ms), 17.1x at 16k (1,320 -> 77 ms), 22 GB/s (01-evidence-ledger-dexa.md, lmcache-offload-restore). Caveats: A100, vLLM 0.24.0, LMCache version unpinned, prefix caching off, single request.
- fp8 (e4m3) and int8 KV round-trip at 0.5x bytes with 100% greedy agreement (kv-interchange-formats). Caveats: HF-level, one 256-token passage, 48-token horizon, Llama-3.1-8B; not vLLM's FP8 KV path.

### Bounded / contradicted

- **1.5x, single seed, 8 ms margin (BOUNDED).** D N=300 p95 272.2 ms vs B-voice 316.8, p99 417.7 vs 482.7, p50 unchanged (98.4 vs 100.7), same 27,338 turns. Full-load window: 291.9 ms (n=21,866), an 8.1 ms pass; 3 of 7 full-load 5-min buckets exceeded 300 ms (B-voice 6 of 7); the docs report whole-run only. N=400 with the 10 GB budget collapsed within 10 min. Budget, hold multiplier and 25 s threshold are untuned first guesses; the hint uses the trace's own medians, so real-call predictor accuracy is unmeasured.
- **Connector stability at N=300 (BOUNDED).** A ~600-line connector kept N=300 stable (p50 100.7 ms, 713 tok/s, run 3,240 s) where LMCache 0.3.2 collapsed (p95 25,997.7 ms default, 6,025.1 ms layerwise). The 82x/19x tail ratio compares a collapsed and a stable system differing in chunking, eviction, decode-KV handling and maturity, on a 2025 LMCache; it says nothing about LMCache 0.5.x or the OffloadingConnector.
- **Latency tax below the wall.** At N=200, B-voice p95 117.5 ms vs baseline 93.1 (+24.4 ms; LMCache +77.5): synchronous restore is on the TTFT path.
- **Proactive turn-end offload was never built.** RESULTS.md's ~4 GB working-set projection is unvalidated (conflict 9): tested is cheap eviction plus pinning, not a bounded working set.
- **Duty cycle 95.2% is synthetic** (conflict 13); speech-to-speech models could collapse the window.
- **Arm C archived-run integrity (CONTRADICTED).** N=300 p95 14,958 ms is over 17,177 of 27,338 turns; the server was unreachable from minute 41.1 (10,146 ConnectionRefused); which attempt was archived is unverifiable. Collapse is visible from minute 5-10, so the sign stands: request-level prefetch fails at the wall.
- **"7 ms restore" is a docstring estimate**; server-side hit rates (35% -> 45%, 8,000+ restores) are unarchived (discrepancies 7, 12). **Audio signals** (VAD/endpoint) contributed nothing as a prefetch trigger; D uses none; untested as a residency signal (vi-audio-signal-contribution).
- **Cross-repo contradiction on restore-under-load.** Dexa's disk-backed connector lost to vLLM re-prefill (0.37x at 8B/8k; P99 8,930 vs 5,596 ms at 24-way); pinned-RAM restores won 10-17x single-request; conflict 1 reconciles by tier bandwidth (~0.6-0.9 vs 10-20 GB/s). RAM-tier restore under concurrency is measured only for 2.5k-token sessions.
- **Shared-prefix eviction bug (code reading).** A store evicts the entry it matched, so a new session's first turn evicts a sibling sharing its system prompt; with 5 customers across 300 sessions the slab holds one entry per prefix lineage; hit-rate effect unquantified (03-build-inventory.md).

### Unproven

Costs are mine, from archived run durations (N=300 ~3,240 s; N=400 5,374 s) and the docs' "~2 GPU-hours per idea".

| Claim | Experiment that proves or kills it | Est. cost |
|---|---|---|
| Result survives an 80 GB GPU (docs' #1 threat) | Arm A and D on H100, 7B, N=300/400/500; conflict 5 arithmetic: a ~60 GB pool fits the N=300 working set and moves A's cliff to ~N 400-450 (not run). Today's code supports --gpu H100 | ~6 GPU-h, 2 days |
| A current stock tier (vLLM 0.28 OffloadingConnector lru/arc; LMCache 0.5.4 layerwise; SGLang HiCache) does not already hold N=300 | Each on identical traces | ~16 GPU-h, 5 days |
| D reaches >= 2x at 3 seeds | Seeds 1-3, A/B-voice/D, N=200/250/300/350/400; budget 10 -> 16 GB; hold sweep | ~30 GPU-h, 5 days |
| The predictor, not just pinning, carries the gain | D with shuffled idle_ms vs real hints; pin precision logged | ~6 GPU-h, 2 days |
| Proactive free bounds the working set | Free-after-save for predicted long-idle sessions; log HBM pool occupancy | ~6 GPU-h, 3 days |
| Holds on a production-size model | Gemma 4 31B or Qwen3.6-35B-A3B on H100, fp16 and FP8 KV; needs TP>1 and hybrid-attention layout (new code) | ~20 GPU-h, 7 days |
| Idle windows exist on real calls | Transcript -> SessionTrace converter; Phase 0 at 45 tok/s, 57,344 B/token; gates <40% STOP, 40-60% 2x ceiling, >60% proceed | 5 days + partner transcripts |
| Restore is bit-correct; restore latency measured | Run and archive evals/modal_kv_correctness.py; CUDA-event timing per restore | ~3 GPU-h, 2 days |
| Shared-prefix bug neither flatters nor hurts D | Fix eviction; --shared-prompt arm | ~6 GPU-h, 2 days |
| Speech-to-speech survives | Ultravox-class model on the harness | unmeasured; out of MVP |

## 7. MVP and 6-week build plan

**Ships first (the draft's choice; reopened in section 11):** a BYOC package — connector plugin + session gateway + harness — on one design partner's GPUs, reporting sessions-per-GPU at p95 <= 300 ms against their own vanilla vLLM with raw logs. No design partner is named anywhere.

- **Week 1 — kill tests before the port.** On the existing vLLM 0.9.2 code: (a) arm A on H100 at N=300/400 via evals/modal_arm_a.py --gpu H100; (b) vLLM-native OffloadingConnector and LMCache 0.5.4 on L40S at N=300 (config only). In parallel, start porting voice_kv_connector.py to 0.28.0 (experimental API; expect drift). Reuse: vkv/backends/voice_kv_connector.py; dexa's async-load pattern (src/dexa/engine/vllm_connector.py). Fix the shared-prefix eviction.
- **Week 2 — port, correctness, timing.** Async/layer-pipelined loads; run and archive the correctness suite; CUDA-event restore timing; server-metrics scraping; connector/config provenance in manifests. Reuse: vkv/metrics, evals/modal_arm_a.py.
- **Week 3 — replicate and ablate on L40S/7B.** A/B-voice/D at 3 seeds, N=200/250/300/350/400; shuffled-hint ablation; 16 GB budget; hold sweep; proactive-free arm; full-load-window and multi-seed reporting (new). Kill-gate check.
- **Week 4 — production model.** Gemma 4 31B or Qwen3.6-35B-A3B on H100 (TP>1: new); FP8 KV; SGLang baseline; --shared-prompt arm. Transcript converter (vkv/traces/schema.py) if a partner supplies transcripts.
- **Week 5 — gateway.** Streaming SSE (app.py forces stream=False today), Retell WebSocket, session -> replica affinity map (new), idle_ms injection, keys/metering (dexa_platform/control).
- **Week 6 — package and publish.** BYOC compose (pattern: dexa_platform/docker-compose.byoc.yml), partner deployment, benchmark report with raw events.jsonl.

**Slip assessment.** Weeks 1-2 carry the API port (unknown drift), week 4 carries TP>1 and hybrid-attention layout for a 30B model (new code), week 6 depends on a partner that does not exist yet. A credible public proof (3 seeds, H100, 30B-class model, current stock baselines, one real-transcript trace set) is realistically 10-14 weeks.

## 8. Pricing model

Candidate unit (founder decision): **per concurrent session-hour (a "line")** against a published sessions-per-GPU figure at p95 <= 300 ms; BYOC as a per-GPU software fee against the same figure, mirroring Vapi ($10/line/month) and Retell ($8/concurrency/month). Expressible only if the capacity figure replicates and predictor precision holds on real calls. Per-token providers do not price idle residency: frontier caches expire at 5 min default (Anthropic), 30 min (OpenAI GPT-5.6+), or "may be evicted at any time" (xAI); no first-party provider was found selling guaranteed pinned KV (05-research-caching.md; 05-research-agents.md, Unknowns). Nearest priced primitives: DeepInfra's explicit 5m/1h retention at 1.25x/2.0x to write, 0.2x to read (two models); Fireworks' x-session-affinity, ~50% cached discount; Tensormesh cached input at $0; Anthropic Managed Agents $0.08/session-hour while running.

Cost floor (arithmetic, not measurement): Modal L40S $1.95/hr plus $0.00799/GiB-hr memory, so the 96 GiB host RAM the tiered arms used adds ~$0.77/hr (~40%), $2.72/hr total; bare-metal clouds bundle host RAM (RunPod H100 SXM 125 GB/GPU; Nebius 200 GB/GPU) (05-research-gpu-pricing.md). At 300 sessions GPU+RAM is ~$0.0091 per session-hour vs ~$0.0136 at 200 ($0.0065 vs $0.0098 GPU-only); Vapi's $10/line/month is ~$0.0137/line-hour at 730 h. L40S/7B numbers only.

## 9. Competitive facts

| Who | Adjacent thing they ship | Not shipped, as far as the research shows | Source |
|---|---|---|---|
| LiveKit | Gemma 4 31B on SGLang + spec decode, 192 ms TTFT; "provisioned LLM capacity"; concurrency 5/20/50 | Sessions-per-GPU figure; conversation-aware residency | 05-research-voice.md |
| Vapi | Custom-LLM via OpenAI-compatible SSE; $10/line/mo; 1-5M calls/day | Hosting its own models | 05-research-voice.md |
| Retell | Custom-LLM WebSocket; $8/concurrency/mo; LLM per minute $0.003-$0.16 | Self-hosted LLM disclosure | 05-research-voice.md |
| ElevenLabs / Inworld / Deepgram | Self-hosted open models (Qwen3.6-35B-A3B; Gemma 4 26B; Nemotron-3-nano-30B); BYO LLM | Sessions-per-GPU; residency policy | 05-research-voice.md |
| Wafer | Dedicated endpoints tuned across engine/kernels/hardware; logos include Vapi, Tavus, Inworld; Neon Health TTFT 800 -> ~550 ms; $40M Series A | Session/KV residency; sessions-per-GPU; public dedicated pricing | 05-research-wafer.md |
| Fireworks / DeepInfra / Baseten | x-session-affinity, ~50% cached discount, BYOC private preview; explicit 5m/1h retention at 1.25x/2x (two models); KV-aware routing | Idle-window prediction; sessions-per-GPU | 05-research-providers.md |
| Tensormesh / LMCache | KV tiering CPU/disk/remote, LRU-family watermark eviction; cached input at $0 | TTL/idle-window-aware eviction (none documented) | 05-research-kvcache.md |
| vLLM OffloadingConnector | CPU/FS/S3/P2P tiers, lru/arc, custom policy hook; TTFT 2-22x on 8B/H100 | A session residency policy | 05-research-kvcache.md |
| SGLang | HiCache L1/L2/L3; RFC #27574 Pin with bounded TTL (open); Q3 "session-aware RadixTree" roadmap | Shipped session-aware pinning | 05-research-kvcache.md |
| Mooncake | Soft-pin TTL (default 30 min), hard-pin; draft PR #2835 TTL leases | Predictive (not client-fixed) TTL | 05-research-kvcache.md |
| Continuum (paper) | TTL-pinned GPU KV from reload cost; >8x JCT on agent benchmarks | Voice workload; product | 05-research-agents.md |
| Morph | Sticky KV placement via x-session-id; per-request cache TTL control; Standby tier; Reflexes engine "forked from vLLM" | Voice serving | 05-research-morph.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| 80 GB GPU dissolves the wall (docs' #1 threat) | Arm A on H100, 7B, N=300/400/500 (week 1) | A passes N >= 400: L40S/7B framing dead; re-scope to 30B-on-H100 or stop | A's H100 knee < 400 |
| A stock offload tier already holds the load | vLLM 0.28 OffloadingConnector / LMCache 0.5.4 at N=300 (week 1) | Stock tier p95 <= 300 ms at N=300: differentiation shrinks to the pin policy; re-base gates on D vs stock | Stock tier > 300 ms or collapses |
| 1.5x does not replicate or reach 2x | Median max passing N, seeds 1-3, full-load-window p95 (every seed must pass at the claimed N), after tuning | D/best-baseline < 1.5x, or any seed > 300 ms; 1.5-2x conditional on FP8 KV (founder decision) | >= 2x, all seeds <= 300 ms |
| Predictor carries no information | Shuffled-hint ablation | Real hints <= random within seed spread | Real beats random by >= 20 ms p95 at the knee |
| API port breaks pin/delayed-free | Correctness suite + N=5 smoke on vLLM 0.28 | Delayed-free unsupported, or any bitwise mismatch / 'missing entry' | Bitwise pass, zero errors |
| 30B model / TP>1 / hybrid attention | D vs A on H100, Gemma 4 31B or Qwen3.6-35B-A3B | D/A < 1.3x or unsupported | D/A >= 1.8x |
| Real calls lack idle windows | Phase 0 on transcript traces, 45 tok/s, real KV geometry | Idle < 40% (pre-registered STOP) | > 60% |
| Predictor is wrong on real calls | Pin precision (pinned sessions returning within hold) | < 50% | > 80% |
| SGLang already delivers the capacity | SGLang HiCache baseline N | SGLang N >= D N | SGLang N <= A N x 1.2 |
| Shared prompts flatter baseline / eviction bug | --shared-prompt arm after the fix | D/A < 1.3x | D/A >= 1.5x |
| Below-wall latency tax | B-voice/D vs A p95 at N=100-200, async loads | > +40 ms p95 | <= +25 ms p95 |

## 11. Founder decisions

- **Hosted vs BYOC vs open-source connector.** The draft chose BYOC-first; reopened here. Options: hosted on own GPUs; BYOC on platform GPUs (LiveKit/Inworld self-host); OSS the connector, sell gateway plus benchmark. Informed by whether platforms will run a third-party plugin.
- **Which kill test runs first.** The plan runs the H100 and stock-tier tests in week 1 (cheapest, most lethal); the founder may prefer to port first.
- **Engine.** vLLM (plugin exists) vs SGLang (LiveKit's engine; RFC #27574 in flight) vs both.
- **The SLA number.** 300 ms p95 (repo PRD) vs Twilio's 375/750; a looser SLA moves every knee.
- **The ratio worth building on.** 1.5x with "no cliff" vs 2x vs 3x; the repo's own thresholds conflict (conflict 8).
- **Whether adjacency matters.** Wafer lists Vapi, Tavus, Inworld and raised $40M; Tensormesh sells KV tiering; SGLang has an open pin-with-TTL RFC; Mooncake soft-pin TTLs; Morph per-request TTL. None ships predicted-idle residency or a sessions-per-GPU number; durability of the gap is not our call.
- **Upstream or proprietary policy; leaderboard timing.** vLLM's OffloadingConnector exposes a custom policy hook: upstream (distribution) or keep (moat). Publishing the harness first invites replication by engine maintainers; later delays the proof.
- **Cost product or latency product.** LLM share of a voice minute: <5% on an Inworld-style stack, up to ~65% on Retell's frontier lines.
- **Model, GPU class, serverless vs bare metal.** Evidence is 7B fp16 on Modal L40S ($0.79-$2.25/hr across clouds; ~40% RAM surcharge on Modal); production defaults are 26-35B; H100 $1.73 (aggregator) to $12.29/hr (05-research-gpu-pricing.md).
- **Fork vLLM for scheduler-native prefetch** (docs' "strongest remaining version", untested) vs stay a plugin.
- **Speech-to-speech bet.** Unmeasured; may collapse the idle window.
- **Pricing unit.** Per line-hour, per GPU-hour with a guarantee, or per token with a pinned-session surcharge.
- **Platform or its customers.** Platforms buy capacity; customers buy an endpoint.

## 12. Combinations

- **With session-persistent inference for agents (dexa stateful sessions):** same connector and gateway; agent idle gaps are tool-driven (5.2 s median, 81.4 s P99, Codex traces) or human (1.4 min median, p90 20.6 min), so only the predictor changes (05-research-agents.md).
- **With a Wafer-style engine/kernel/hardware tuning layer:** orthogonal; residency sits above whatever engine configuration is fastest.
- **With the benchmark harness as a standalone product:** a public "sessions-per-GPU at p95 <= 300 ms" leaderboard; none found (search incomplete).
- **With a Morph-style specialized voice model + speculator:** a later layer; the connector is model-agnostic.
