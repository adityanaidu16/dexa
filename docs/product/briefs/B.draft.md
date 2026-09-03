# Candidate B — Conversation-native inference for voice agents

## 1. Thesis

A voice-agent LLM inference company whose product metric is **concurrent voice sessions per GPU at p95 time-to-first-token (TTFT) <= 300 ms**, sold to voice platforms and their customers as a hosted OpenAI-compatible endpoint and as a bring-your-own-cloud (BYOC) package. The mechanism is session-semantic KV residency: each request carries the conversation's predicted idle window; the serving stack pins sessions predicted to return soon out of the engine's evictable pool and lets the rest be evicted by the engine's LRU into a contiguous pinned-CPU restore path, so eviction becomes a cheap load rather than a re-prefill. It is a vLLM KVConnector plugin plus a session gateway; no fork. The customer's sentence: "We know exactly how many calls a GPU holds at p95 under 300 ms, and it doesn't fall off a cliff when we add lines."

## 2. Customer and workload

**Who buys.** (a) Voice platforms that already self-host an open-weight LLM: LiveKit (Gemma 4 31B on SGLang with speculative decoding, 192 ms TTFT, $0.40/$1.20 per 1M tokens), Inworld (Gemma 4 26B), ElevenLabs (Qwen3.6-35B-A3B, Qwen3.5-397B-A17B), Deepgram (Nemotron-3-nano-30B) (05-research-voice.md, Summary and Facts). (b) Their customers who bring their own LLM: every orchestrator fetched except Bland accepts a custom LLM over OpenAI-compatible SSE (Vapi, ElevenLabs) or WebSocket (Retell) (05-research-voice.md).

**What they run today.** Cascaded STT -> LLM -> TTS; each turn resubmits the full history; the LLM is a 26-35B open model or a frontier API. The measured workload in the voice-inference repo is smaller: Qwen2.5-7B fp16, ~2.5k tokens per session by turn end, ~140 MB KV per session at 57,344 B/token, median reply 75 tokens (02-evidence-ledger-voice.md, header and vi-armA-baseline-knee).

**Turn and idle pattern.** Decode finishes far ahead of speech playback (measured ITL ~22 ms, ~45 tok/s, vs 3.25 tok/s speech). On synthetic traces the model is compute-idle 95.2% of wall clock and >93% of idle windows exceed 8 s (vi-phase0-duty-cycle, BOUNDED: all timing distributions assumed).

**Concurrency and payment.** Concurrency is sold as lines: Vapi $10/line/month beyond 10; Retell $8/concurrency/month beyond 20; LiveKit 5/20/50 inference concurrency by plan. The LLM line is paid per token (LiveKit), per minute by model (Retell $0.003-$0.16/min), at cost (Vapi, Inworld), or bundled (Bland $0.11-$0.14/min) (05-research-voice.md). Twilio budgets 375 ms target / 750 ms max for LLM TTFT inside a ~1.1 s turn (05-research-voice.md).

## 3. The pain, in the customer's words

- LiveKit, on its own Gemma 4 deployment: it reserves "more headroom than a throughput-maximized deployment would" to minimize queueing, "accepting higher per-request cost" (05-research-voice.md, LiveKit blog 2026-07-02).
- Neon Health (a voice agent, on Wafer): "it doesn't go off a cliff when you increase the requests per minute" (05-research-wafer.md, homepage testimonial). The repo measured the cliff: baseline vLLM serves 200 synthetic sessions at p95 93.1 ms and then collapses (not degrades) between 200 and 300 as prefix-cache hit rate falls from ~92% to 1.6-1.9% (vi-armA-baseline-knee, PROVEN).
- Deepgram: "You'll pay the LLM separately on both pass-through models, and it's usually the largest line." Inworld's model puts a self-hosted LLM under 5% of a minute (TTS 70-73%); smallest.ai's example ~30% ($0.020 of $0.066); Retell's per-model LLM line spans ~4% to ~65% (05-research-voice.md). The pain is TTFT and capacity for some stacks, cost for others.
- No vendor advertises sessions-per-GPU for the LLM stage; the only per-GPU concurrency figures found are Kyutai STT (64 streams/L40S, 400/H100) and Rime TTS (100+/machine) (05-research-voice.md, Unknowns).

## 4. Value proposition and the proof-of-value benchmark

**Metric.** Max concurrent sessions N at p95 TTFT <= 300 ms, TTFT measured client-side from STT-final to first streamed chunk, reported over the **full-load window** (minutes 5-40) as primary and whole-run as secondary, median and spread over 3 seeds, runs >= 40 min because collapse was depth-triggered at minute 15-20 and a 10-minute test would pass a failing config (vi-armA-baseline-knee caveats; vi-harness-methodology-lessons).

**Setup.** The voice-inference replay harness: trace-driven sessions with turn-taking, barge-in (10%), spurious VAD (5%), full-history resubmission, 60 s warmup discard, archived events.jsonl per point (03-build-inventory.md). Traces: synthetic generator plus a real-transcript converter (unbuilt; vi-open-real-transcript-traces).

**Baselines, by name.** Vanilla vLLM with prefix caching (arm A); vLLM + LMCache CPU offload, default and layerwise (arm B-LMCache); SGLang RadixAttention + HiCache host tier (unrun); our connector without pinning (B-voice) and with pinning (D). Hosted endpoints (Fireworks, LiveKit) compare on TTFT only; sessions-per-GPU is not observable.

**Measured so far (seed 1, L40S 46,068 MiB, Qwen2.5-7B fp16, vLLM 0.9.2).** Max passing N: A 200, B-LMCache 200, B-voice 200, C 200, D 300 -> 1.5x (RESULTS.md Headline; sweep_d/results.json). At N=300: LMCache default p95 25,997.7 ms; B-voice 316.8 ms stable; D 272.2 ms; D at N=400 34,048 ms (vi-armD-predictive-pinning; vi-armD-n400-failure).

**Target number.** >= 2x vanilla vLLM's max passing N on the same GPU and model, median of 3 seeds, full-load-window p95. The repo's pre-registered README table reads "< 2x kill · 2-3x conditional (try FP8 KV) · > 3x proceed", so today's 1.5x sits in its own kill band; RESULTS.md's "above arm B's kill gate (>= 1.5x), below the 2x proceed line" is a post-hoc re-scoring against the Phase-2 gate (04-conflicts.md, conflict 8). Secondary target: no collapse — p95 under 1 s at 1.5x the baseline knee.

**Why a skeptical buyer would believe it.** Harness and raw logs are published (287,015 turns already archived across 19 runs); baselines are the buyer's own stack; the plugin runs inside the buyer's vLLM, so they reproduce on their GPUs and transcripts; the restore path ships with an archived bitwise correctness result (the suite exists; no output is committed; vi-kv-restore-correctness). Not claimable until measured: "2x", "robust across seeds", "on H100", "on 30B models", "on real calls".

## 5. Architecture

| Component | Custom / OSS | Role |
|---|---|---|
| Session KV connector | Custom (vLLM KVConnectorBase_V1 plugin, no fork) | Save block-aligned prompt KV to one pinned-CPU slab with one async D2H copy per request; content-addressed prefix chain (blake2b over 16-token blocks); one contiguous H2D + scatter on restore; delayed-free pin via request_finished/get_finished (voice-inference/vkv/backends/voice_kv_connector.py, 574 lines) |
| Residency policy | Custom | Turn idle_ms hint into pin/hold/evict decisions under a pin budget; today hold = idle x 1.3 + 2,000 ms, pin if idle <= 25 s and the 10 GB budget allows. Proactive free of predicted long-idle sessions: designed, not built |
| Idle-window predictor | Custom | Today a 6-line function: TTS playback remainder + turn-taking medians (vkv/run.py _idle_hint). Later: learned per-customer predictor |
| Session gateway | Custom (from dexa_platform/gateway, 692 lines; edge/ Durable Object pattern) | OpenAI-compatible SSE and Retell-style WebSocket; session-id -> replica affinity; injects kv_transfer_params.idle_ms; tenant keys and metering (dexa_platform/control) |
| Engine | OSS: vLLM (target 0.28.0); SGLang HiCache as second target | Unmodified scheduler, attention, kernels |
| Benchmark harness | Custom (vkv orchestrator/loadgen/metrics/sweep, ~2,000 lines) | The proof-of-value instrument; published |
| Correctness suite | Custom (evals/modal_kv_correctness.py) | Bitwise slab round-trip + forced-evict restore probe |
| Server telemetry | New | Prefix-hit rate, queue depth, restore latency into events.jsonl (today uncommitted) |

**Where the differentiation lives:** the connector's memory policy and the gateway that carries conversational state into it — the layer between engine and orchestrator; not the model, kernels or scheduler. What is replaced is the residency decision, which is LRU-family in LMCache, vLLM's OffloadingConnector and Mooncake (05-research-kvcache.md).

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

- Baseline vLLM concurrency cliff on L40S/7B: N=200 passes at p95 93.1 ms (18,324 turns); N=300 collapses; prefix hit 92% -> 1.6-1.9% (vi-armA-baseline-knee; runs/modal/sweep_a).
- Predictive pinning is the only configuration of five that moved the knee: D N=300 p95 272.2 ms vs B-voice 316.8 ms, p99 417.7 vs 482.7, same 27,338 turns; p50 unchanged (98.4 vs 100.7) (vi-armD-predictive-pinning; runs/modal/sweep_d/seed1/n300/summary.json).
- A ~600-line connector kept N=300 fully stable (p50 100.7 ms, 713 tok/s, run 3,240 s) where LMCache 0.3.2 collapsed (p95 25,997.7 ms default, 6,025.1 ms layerwise; run 4,244 s). The 82x/19x ratio compares a collapsed and a stable system that differ in chunking, eviction and maturity; not a per-transfer claim (vi-armB-voice-connector-reactive; vi-contiguous-vs-chunked-transfer).
- CPU restore beats re-prefill single-request in a real vLLM stack: LMCache 10.0x at 4k (301 -> 30 ms), 17.1x at 16k (1,320 -> 77 ms), 22 GB/s (01-evidence-ledger-dexa.md, lmcache-offload-restore).
- fp8 (e4m3) and int8 KV round-trip at 0.5x bytes with 100% greedy agreement over 48 tokens (01-evidence-ledger-dexa.md, kv-interchange-formats).

### Bounded / contradicted

- **1.5x, not 2x, single seed, in the README's kill band.** D's pass margin is 27.8 ms on whole-run p95 and 8.1 ms on the full-load window (291.9 ms, n=21,866); 3 of 7 full-load 5-min buckets exceeded 300 ms (B-voice 6 of 7); knee bracketed 300 <= knee < 400; N=400 with the 10 GB budget failed at 34,048 ms within 10 min; the docs report whole-run percentiles only (02-evidence-ledger-voice.md, discrepancy 2; vi-armD-n400-failure; 04-conflicts.md conflict 8).
- **The connector costs latency below the wall.** At N=200, B-voice p95 is 117.5 ms vs baseline 93.1 ms (+24.4 ms; LMCache +77.5 ms) — reactive lookup/restore sits on the TTFT path (vi-contiguous-vs-chunked-transfer).
- **Proactive turn-end offload was never built.** RESULTS.md projects a ~4 GB working set at N=300 from copy+free at turn end; the built connector keeps HBM vLLM-managed and only delays frees for pins (04-conflicts.md conflict 9). Tested: cheap eviction plus pinning; untested: bounded working set.
- **Duty cycle 95.2% is synthetic.** Phase 0 assumed decode 60 tok/s and 131,072 B/token; the GPU rig measured ~45 tok/s and 57,344 B/token; not re-run. Speech-to-speech models could collapse the idle window (vi-phase0-duty-cycle; conflict 13).
- **Arm C archived-run integrity (CONTRADICTED).** The N=300 p95 of 14,958 ms is computed over 17,177 of 27,338 turns; the server was unreachable from minute 41.1 (10,146 ConnectionRefused); RESULTS.md calls it post-fix and "not of a bug", but manifests carry git_sha 'unknown', so which attempt was archived is unverifiable. Collapse (p95 ~12-17 s) is visible from minute 5-10, so the sign stands: prefetch-as-request fails at the wall (vi-armC-n300-archived-run-integrity; conflict 14).
- **"7 ms restore" is a docstring estimate** at PCIe Gen4 rates; no transfer timings archived (discrepancy 12). **Server-side hit rates (35% -> 45%, 8,000+ restores) are not archived** (discrepancy 7).
- **Cross-repo contradiction on restore-under-load.** In dexa, raw-KV loads from a Modal Volume lost to vLLM re-prefill (0.37x single request at 8B/8k; P99 8,930 vs 5,596 ms at 24-way concurrency) while pinned-RAM restores won 10-17x single-request; conflict 1 reconciles this by tier bandwidth (~0.6-0.9 GB/s disk vs 10-20 GB/s pinned RAM). No run measures RAM-tier restore under concurrency beyond 2.5k-token sessions.
- **Audio signal.** VAD/endpoint events contributed nothing measurable as a prefetch trigger; D uses none; their value as a residency signal is untested (vi-audio-signal-contribution).
- **LLM share of a voice minute** ranges from under 5% (Inworld) to ~30% (smallest.ai) to ~4-65% (Retell derived) (05-research-voice.md).
- **Code-reading finding:** a store evicts the slab entry it matched, so sessions sharing a customer system prompt evict each other's entries every turn; hit-rate impact unquantified (03-build-inventory.md, voice-inference Gaps).

### Unproven

Cost estimates are mine, derived from archived run durations (N=300 point 3,240 s; N=400 5,374 s) and the docs' "~2 GPU-hours per idea" (vi-open-armD-tuning).

| Claim | Experiment that proves or kills it | Est. cost |
|---|---|---|
| D reaches >= 2x at 3 seeds | Seeds 1-3 for A, B-voice, D at N=200/250/300/350/400; pin budget 10 -> 16 GB; hold multiplier sweep | ~25 GPU-h, 4 days |
| Proactive free bounds the working set | request_finished delayed-free + explicit free-after-save for predicted long-idle sessions; log HBM pool occupancy; N=300/400 | ~6 GPU-h, 3 days |
| Result survives an 80 GB GPU | Same sweep on H100; conflict 5 arithmetic: a ~60 GB pool fits the N=300 working set and moves A's cliff to ~N 400-450 (not run) | ~12 GPU-h, 2 days |
| Result holds on a production-size model | Gemma 4 31B or Qwen3.6-35B-A3B on H100, fp16 and FP8 KV | ~20 GPU-h, 5 days |
| Idle windows exist on real calls | Transcript -> SessionTrace converter; rerun Phase 0 at 45 tok/s with real KV geometry; gates <40% STOP, 40-60% 2x ceiling, >60% proceed | ~0 GPU-h, 5 days + partner transcripts |
| SGLang HiCache does not already reach D's N | SGLang RadixAttention + host tier on identical traces | ~8 GPU-h, 3 days |
| Restore is bit-correct | Run evals/modal_kv_correctness.py; archive output | ~1 GPU-h, 1 day |
| Restore latency (replace the 7 ms estimate) | CUDA-event timing per restore into events.jsonl | ~2 GPU-h, 2 days |
| FP8 KV compounds with pinning | vLLM FP8 KV on the same rig | ~4 GPU-h, 1 day |
| Speech-to-speech survives | Ultravox-class model on the harness | unmeasured; out of MVP |

## 7. MVP and 6-week build plan

**Ships first:** a BYOC package — vLLM connector plugin + session gateway + benchmark harness — deployed on one design partner's GPUs, reporting their sessions-per-GPU at p95 <= 300 ms against their own vanilla vLLM with raw logs. The hosted endpoint follows on the same code.

- **Week 1 — connector port and correctness.** Port voice_kv_connector.py from vLLM 0.9.2 to 0.28.0 (the API is experimental and has drifted); make loads async/layer-pipelined (today synchronous in the scheduler step); run and archive the correctness suite; add CUDA-event restore timing. Reuse: vkv/backends/voice_kv_connector.py; dexa's async-load pattern (src/dexa/engine/vllm_connector.py). New: TP>1, MLA layout, proactive free.
- **Week 2 — harness generalization.** Text prompts + tokenizer + chat endpoint, aiohttp mid-stream abort, server-metrics scraping, config provenance in manifests, multi-seed aggregation, full-load-window reporting; transcript converter to SessionTrace JSONL (vkv/traces/schema.py). Reuse: vkv/orchestrator, loadgen, metrics, sweep; evals/modal_arm_a.py.
- **Week 3 — replicate and tune.** A/B-voice/D at 3 seeds on L40S; N=250/350; 16 GB budget; hold sweep; proactive-free arm. Kill-gate check (Section 10).
- **Week 4 — move up.** H100 80 GB; 30B-class model; SGLang baseline; FP8 KV; shared-prompt arm.
- **Week 5 — gateway.** Streaming SSE (dexa_platform/gateway/app.py forces stream=False today), Retell WebSocket, session-id affinity, idle_ms injection, keys/metering from dexa_platform/control. New: session -> replica map.
- **Week 6 — package and publish.** BYOC compose (pattern: dexa_platform/docker-compose.byoc.yml), partner deployment, benchmark report with raw events.jsonl.

## 8. Pricing model

Billed **per concurrent session-hour (a "line")** against a published sessions-per-GPU figure at p95 <= 300 ms; BYOC as a per-GPU software fee against the same figure. This mirrors Vapi ($10/line/month) and Retell ($8/concurrency/month). The architecture makes it expressible because capacity is a residency-policy output the provider controls and measures. A per-token provider's unit cannot price idle residency: frontier caches expire at 5 min default (Anthropic), 30 min (OpenAI GPT-5.6+), or "may be evicted at any time" (xAI); no first-party provider sells guaranteed pinned KV (05-research-caching.md; 05-research-agents.md). The nearest priced primitives: DeepInfra's explicit 5m/1h cache retention at 1.25x/2.0x input to write and 0.2x to read; Fireworks' x-session-affinity with ~50% cached discount; Anthropic Managed Agents at $0.08/session-hour while running (05-research-providers.md; 05-research-caching.md).

Cost floor, derived from published rates (arithmetic, not a measurement): Modal L40S is $1.95/hr and container memory $0.00799/GiB-hr, so the 96 GiB host RAM the tiered arms used adds ~$0.77/hr, 40% on top of the GPU; bare-metal clouds bundle host RAM (RunPod H100 SXM 125 GB/GPU; Nebius 200 GB/GPU) (05-research-gpu-pricing.md). At 300 sessions on a $1.95/hr L40S the GPU cost is ~$0.0065 per session-hour vs ~$0.0098 at 200; Vapi's $10/line/month is ~$0.0137/line-hour.

## 9. Competitive facts

| Who | Adjacent thing they ship | Not shipped, as far as the research shows | Source |
|---|---|---|---|
| LiveKit | Gemma 4 31B on SGLang + spec decode, 192 ms TTFT; inference concurrency 5/20/50 | Sessions-per-GPU figure; conversation-aware residency | 05-research-voice.md |
| Vapi | Custom-LLM via OpenAI-compatible SSE; $10/line/mo; 1-5M calls/day | Hosting its own models | 05-research-voice.md |
| Retell | Custom-LLM WebSocket; $8/concurrency/mo; LLM per minute $0.003-$0.16 | Self-hosted LLM disclosure | 05-research-voice.md |
| ElevenLabs / Inworld / Deepgram | Self-hosted open models (Qwen3.6-35B-A3B; Gemma 4 26B; Nemotron-3-nano-30B); BYO LLM | Sessions-per-GPU; residency policy | 05-research-voice.md |
| Wafer | Dedicated endpoints tuned across engine/kernels/hardware; logos include Vapi, Tavus, Inworld; Neon Health TTFT 800 -> ~550 ms | Session/KV residency; sessions-per-GPU; public dedicated pricing | 05-research-wafer.md |
| Fireworks / DeepInfra / Baseten | x-session-affinity, ~50% cached discount, BYOC private preview; explicit 5m/1h cache retention priced 1.25x/2x; KV-aware routing | Idle-window prediction; sessions-per-GPU; voice-specific residency | 05-research-providers.md |
| Tensormesh / LMCache | KV tiering CPU/disk/remote, LRU-family watermark eviction; cached input at $0 | TTL/idle-window-aware eviction (none documented) | 05-research-kvcache.md |
| vLLM OffloadingConnector | CPU/FS/S3/P2P tiers, lru/arc, custom policy hook | A session residency policy | 05-research-kvcache.md |
| SGLang | HiCache L1/L2/L3; RFC #27574 Pin with bounded TTL (open); Q3 "session-aware RadixTree" roadmap | Shipped session-aware pinning | 05-research-kvcache.md |
| Mooncake | Soft-pin TTL (default 30 min), hard-pin; draft PR #2835 TTL leases | Predictive (not client-fixed) TTL | 05-research-kvcache.md |
| Continuum (paper) | TTL-pinned GPU KV from reload cost; >8x JCT on agent benchmarks | Voice workload; product | 05-research-agents.md |
| Morph | Sticky KV placement via x-session-id; Standby tier; engine "forked from vLLM" | Voice serving | 05-research-morph.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| 1.5x does not replicate or reach the README line | Median max passing N, seeds 1-3, full-load-window p95, L40S/7B, after tuning | D/A median < 2x (README table; today's 1.5x is here) | >= 2x (2-3x conditional on FP8 KV per README; > 3x unconditional) |
| Full-load window fails where whole-run passes | Full-load-window p95 at claimed N, each seed | > 300 ms in any seed | <= 300 ms in all seeds |
| 80 GB GPU dissolves the wall (docs' #1 threat) | D/A on H100 with a 30B model | D/A < 1.3x | D/A >= 1.8x |
| Real calls lack idle windows | Phase 0 on transcript traces, 45 tok/s, real KV geometry | Idle < 40% (pre-registered STOP) | > 60% |
| Predictor is wrong on real calls | Pin precision (pinned sessions that returned within hold) | < 50% | > 80% |
| Restore corrupts KV | Correctness suite | Any bitwise mismatch or 'missing entry' | Zero of both |
| SGLang already delivers the capacity | SGLang HiCache baseline N | SGLang N >= D N | SGLang N <= A N x 1.2 |
| Shared prompts flatter baseline / evict-each-other bug | --shared-prompt arm | D/A < 1.3x | D/A >= 1.5x |
| Below-wall latency tax | B-voice/D vs A p95 at N=100-200 | > +40 ms p95 | <= +25 ms p95 |

## 11. Founder decisions

- **Hosted vs BYOC vs open-source connector.** Hosted on own GPUs; BYOC on platform GPUs (LiveKit/Inworld already self-host); OSS the connector and sell gateway plus benchmark. Informed by whether platforms will run a third-party plugin inside their vLLM.
- **Whether adjacency matters.** Wafer lists Vapi, Tavus, Inworld as logos and has a voice TTFT case study; Tensormesh sells KV tiering; SGLang has an open pin-with-TTL RFC; DeepInfra prices cache retention. None ships session-semantic residency or a sessions-per-GPU number; whether that gap is durable is not our call.
- **Cost product or latency product.** LLM share of a voice minute is <5% on an Inworld-style stack and up to ~65% on Retell's frontier lines; the buyer decides whether the pitch is "fewer GPUs" or "no cliff at 300 ms".
- **Model choice.** Evidence is 7B fp16; production defaults are 26-35B.
- **GPU class.** L40S evidence only ($0.99-$2.25/hr across clouds); docs call an 80 GB GPU the #1 threat; H100 is $3.29-$12.29/hr (05-research-gpu-pricing.md).
- **Fork vLLM for scheduler-native prefetch** (the "strongest remaining version of the thesis", untested) vs stay a plugin.
- **Speech-to-speech bet.** Continuous audio ingestion may collapse the idle window; unmeasured.
- **Pricing unit.** Per line-hour, per GPU-hour with a guarantee, or per token with a pinned-session surcharge.
- **Platform or its customers.** Platforms buy capacity; customers buy an endpoint URL.

## 12. Combinations

- **With session-persistent inference for agents (dexa stateful sessions):** same connector and gateway; agent idle gaps are tool-driven (5.2 s median, 81.4 s P99 on Codex traces) or human (1.4 min median, p90 20.6 min), so only the predictor changes (05-research-agents.md).
- **With a Wafer-style engine/kernel/hardware tuning layer:** orthogonal; residency sits above whatever engine configuration is fastest.
- **With the benchmark harness as a standalone product:** a published "sessions-per-GPU at p95 <= 300 ms" leaderboard; nobody publishes this today.
- **With a Morph-style specialized voice model + speculator:** a later layer; the connector is model-agnostic.