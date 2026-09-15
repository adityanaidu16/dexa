## 1. Thesis

Candidate J is an inference service that runs long-lived agent sessions on preemptible (spot) GPUs and makes that safe by making every session's KV state portable: written off the GPU continuously, addressed by session, and restorable on any other instance with bit-identical continuation. The buyer is a team running open-weight models behind a coding, computer-use or back-office agent whose sessions last minutes to hours and sit idle for most of that time. The sentence a customer would repeat: "Our agents run on the cheapest GPUs on the market, and when one gets pulled the session comes back on another one in seconds instead of dying." The qualifier this brief carries throughout: the repos prove portability and losslessness at TP=1, prove that restoring from a fast tier beats re-prefill at 4k-16k inside vLLM+LMCache, and also prove that loading raw KV through the Dexa connector loses to vLLM re-prefill at 8k-32k. So the product's value at agent-scale context (~115k-195k tokens per step in public traces) is continuity and fleet capacity, not TTFT, and the number that decides whether the direction works, restore bandwidth from a store that survives the instance, has never been measured.

## 2. Customer and workload

**Who buys.** Infrastructure engineers at companies shipping agents on open weights: coding-agent products (OpenHands recommends Qwen3.6-35B-A3B on vLLM/SGLang; Cursor's Composer 2 started from Kimi K2.5), computer-use products (Browser Use ships a 30B-A3B open model), and enterprises self-hosting an agent harness. Open-weight models carry a majority of OpenRouter tokens by mid-2026, concentrated in coding and agentic workloads (05-research-agents.md).

**What they run today.** vLLM (0.28.0) or SGLang (0.5.18) behind an OpenAI-compatible endpoint; models from ~30B-A3B MoE up to GLM-5.2/Kimi-K3-class MoE at TP=8, where "TP=8 keeps eight copies of the KV cache" (llm-d GLM-5.2 post). Context: TraceLab's ~4,300 Claude Code/Codex sessions show a median prefix of 126,180 (Claude) / 115,584 (Codex) tokens per step with ~860 new input and ~200-250 output tokens; llm-d's 219 production Claude Code sessions show a median request of 195K input / 317 output, with 96% of requests reusing >=90% of input as verbatim prefix. Input:output ~131:1 (vLLM/Mooncake Codex traces). Idle pattern: human gap median 1.4 min, p90 20.6 min (TraceLab); inter-turn pause median 2 s, p99 11.4 min (llm-d); tool gaps 5.2 s median / 81.4 s P99 (vLLM/Mooncake). Turn length: median ~45 s, 99.9th percentile over 45 min (Anthropic). Fleet concurrency per buyer: unmeasured.

**How they pay today.** On-demand GPU-hours or per-token to hosted providers whose caches are per replica (Fireworks: caching "only works within 1 replica", cached tokens at 50% discount). Spot, per 05-research-gpu-pricing.md: AWS p5 H100 $6.88 on-demand vs $2.59 Spot per GPU-hr (~62% off); p6-b200 $14.24 vs $5.24 (~63%); Azure ND H100 $12.29 list vs $2.37 Spot (~81%); Nebius H100 $3.85 on-demand vs $2.15 preemptible, B200 $7.15 vs $3.95; GCP a3-mega $11.62 vs $3.69 (~68%). The discount is class-specific: AWS A100 p4de Spot is ~19% off and L40S g6e ~16% off, and Azure H200 Spot meters equal list. Relative to neocloud on-demand (H100 median $3.39; Vast $1.73), hyperscaler Spot is roughly at, not below, cheap on-demand. Preemption notice: AWS 2 minutes best-effort; GCP 0 s default or 120 s preview; Azure 30 s best-effort; Vast none. Observed GPU-pool interruption rates: not retrievable (AWS publishes <5% to >20% monthly bands, Azure per-hour rates; no GPU rows found).

## 3. The pain, in the customer's words

- "Spot is 60-80% off list on H100/B200, but I can't put a 40-minute agent turn on a box with a 30-second or zero-second notice." (05-research-gpu-pricing.md; Anthropic: 99.9th-percentile turn >45 min.)
- "When a replica dies or traffic overflows, we re-prefill 130k tokens and the user waits." (OpenAI docs: overflow is a cache miss; Fireworks: per-replica cache.)
- "Past ~150K tokens our Codex runs on Azure got cached_input_tokens = 0; 3.28M re-billed tokens, 71% of fresh input, in one 4-turn run." (openai/codex #25604.)
- "The provider moved us from a 1-hour to a 5-minute cache and the bill went up 17%." (anthropics/claude-code #46829: $949 over 119,866 calls.)
- "Nobody sells a session guaranteed to stay warm, so we over-provision on-demand GPUs." (05-research-agents.md: no provider found selling guaranteed session-pinned KV retention.)

## 4. Value proposition and the proof-of-value benchmark

**Value proposition.** For sessions whose context is large enough that re-prefill is the expensive side of the trade, a portable KV tier lets the session live on interruptible capacity: a preemption becomes a restore from a durable tier on another instance, the same code path as an idle-eviction restore. Below that context size the product adds determinism (identical continuation) and off-GPU capacity, not speed.

**What portable KV adds over re-prefilling from the message log, by context size.** The customer's obvious alternative is to resend the transcript to instance B. The conflicts ledger reduces the comparison to one inequality: restore beats re-prefill iff (tier bandwidth available to this request) x (vLLM re-prefill time for this context) > (KV bytes) (04-conflicts.md, conflict 1). At <=16k, measured: the connector loses (0.37x at 8k) and adds only bit-identical continuation, which re-prefill does not guarantee (voice-inference's correctness suite notes that greedy decoding "cascades cached-vs-full-prefill kernel numerics into divergent text"). At 32k-64k: the repo's estimated crossover is "~64k+" for its disk-backed path, never run because chunked-prefill persistence is missing. At 128k: unmeasured on any engine in either repo. Derived requirement, not a measurement: Llama-3.1-8B KV is 131 KB/token, so 128k = ~16.8 GB; vLLM prefills 8k in 617 ms (~13k tok/s) and 128k in 32.68 s in eager mode with prefix caching off; the required durable-tier bandwidth is therefore between ~0.5 GB/s (eager 128k) and ~1.75 GB/s (linear from 8k), with ~0.8 GB/s derived at 64k. Measured reference points: the connector's Modal-Volume path ~0.6-0.9 GB/s (loses at 8k); LMCache from pinned CPU 22 GB/s (wins, but dies with the host); LMCache/CoreWeave report 8.5-10 GB/s per 8-GPU node from object storage (external, not reproduced). fp8/int8 KV halves the bytes with 100% greedy agreement over 48 tokens (proven, one passage). Neither repo's connector saves decode-token KV, so the MVP resumes at turn boundaries and re-prefills captured partial output.

**The benchmark: kill-and-resume.** N concurrent agent sessions replayed from a trace; at a random point instance A is killed (SIGKILL, no drain) and the sessions' next turns route to instance B. Measured: (a) resume TTFT on B, (b) GPU-seconds to resume, (c) `identical_output` versus an unbroken run, (d) sessions sustained per GPU at a p95 gate. Context ladder 8k/32k/64k/128k; Llama-3.1-8B or Qwen2.5-7B at TP=1 first (the only validated regime).

**Baselines, named.** (1) Vanilla vLLM with prefix caching, re-prefilling from the message log on B: what a buyer would build, and the baseline that beat the connector at 8k. (2) vLLM + LMCache S3-compatible backend. (3) vLLM's native OffloadingConnector with an object-store secondary tier. (4) llm-d P2P pull from a surviving peer's CPU tier (published crossover ~8K-12K tokens for GLM-5.2-FP8; 48K-token pull 235 ms vs 1,988 ms recompute on gpt-oss-120b). (5) A hosted per-token provider with session affinity, compared on cost.

**Pre-registered target.** At L=128k on the kill test: resume TTFT on B below baseline (1)'s re-prefill TTFT; restore GPU-seconds below 20% of re-prefill GPU-seconds; `identical_output` true; p95 TTFT under N>=16 concurrent sessions no worse than baseline (1) by more than 1.2x. At L<=16k we publish the loss.

**Why a skeptical buyer believes it.** It is a kill test, not a warm-cache test; baseline (1) is their own fallback; output identity is checked; losses are published next to wins.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| vLLM engine | OSS, unmodified | prefill/decode, paged KV, prefix caching |
| Session KV connector | custom (KVConnectorBase_V1) | continuous write-through of block-keyed KV to the durable tier; async contiguous restore; chunked-prefill save |
| Durable KV tier | OSS stores (S3-compatible / remote NVMe) + custom blob format | holds session KV off-instance; the tier that survives preemption |
| Session registry + router | custom | session id -> KV location and instance; re-placement on kill |
| Preemption handler | custom | on notice (where one exists): flush unsaved blocks, drain, re-route |
| Spot fleet manager | custom | acquires/releases interruptible capacity; price and interruption telemetry |
| Gateway / control plane | custom (dexa_platform, edge) | OpenAI-compatible API with a `session` field, keys, metering |
| Load harness | custom (voice-inference vkv) | trace replay, kill injection, TTFT/GPU-seconds/identity metrics |

Differentiation lives in the connector's save/load path and the session control plane, not in the model, kernels or scheduler. vLLM is built on, not replaced. Because GCP's default notice is 0 s, the design cannot depend on flush-on-notice; the connector writes through continuously and the notice window covers only the last unsaved blocks (derived: 2 min at 1 GB/s is ~120 GB on AWS, 30 s is ~30 GB on Azure). KVConnector V1 has been labeled experimental since April 2025; API drift is a standing cost.

```
 client (agent harness)
   | OpenAI-compatible + session id
   v
 gateway/router ---- session registry (session -> instance, KV location)
   |                        ^
   v                        |
 vLLM on spot instance A    |   preemption notice (0 s / 30 s / 2 min)
   | connector: write-through save |--> preemption handler: flush, drain, re-route
   v                        |
 durable KV tier (object store / remote NVMe)
   ^                        |
   | connector: async restore
 vLLM on spot instance B  <-+   (bit-identical continuation)
```

## 6. Evidence

### Proven

- KV saved by one vLLM process loads in a separate process with prefill skipped and identical output: two Modal containers sharing a Volume, `A_saved=True, B_saw_stored_KV=True, identical_output=True`; 8176/8192 tokens matched on Qwen3-0.6B/A10G (01-evidence-ledger-dexa.md, cross-instance-resume). TP=1, single attention backend, single-step prefill only.
- Connector conforms to vLLM 0.24.0's V1 interface: 10/10 hook signatures (connector-conformance).
- Lossless persist/resume at 8k-64k on Llama-3.1-8B/A100: `identical_output` true at every length; state 1.0/2.1/4.2/8.4 GB. The speedups in that run are against HF eager prefill and are not cited here (04-conflicts.md, superseded list).
- Fast-tier restore beats re-prefill in real vLLM at small context: LMCache CPU offload/restore 301->30 ms at 4k (10.0x), 1,320->77 ms at 16k (17.1x); 16,384 tokens moved in 39.8 ms (lmcache-offload-restore). HF tensor-copy restore 11.8x at 4k to 28.9x at 64k vs HF prefill; NVMe restore 1.2x-7.0x (stateful-warm-session-hf).
- End-to-end session service demo: 12k-token session, turn 1 cold ~5.4 s, turn 2 warm ~1.5 s (stateful-session-service-live; no raw artifact in repo).
- fp8/int8 KV at 0.5x bytes: 100% greedy agreement over 48 tokens; int4 diverges at token 14 (kv-interchange-formats). KV is keyed to exact weights: base<->instruct injection diverges within 2-5 tokens (kv-interchange-weights).
- A session-granular contiguous-transfer connector held 300 voice sessions on an L40S (p95 316.8 ms) where LMCache 0.3.2's tested configs collapsed (25,997.7 ms) (02-evidence-ledger-voice.md); connector staging buffers outside vLLM's memory budget OOM'd the engine at 345 concurrent requests (vi-bypass-governor-policy).

### Bounded / contradicted

- Raw-KV load through the Dexa connector loses to vLLM re-prefill: 8B/A100/8k resume 1681 ms vs 617 ms (0.37x); under 24x3072-token concurrency async P99 8930 vs 5596 ms; 16x8192 P99 20296 vs 9379 ms; the adaptive policy only matches baseline (5603/9371 ms). Contradiction with the 10-34x entries: different store medium (Modal Volume ~0.6-0.9 GB/s vs pinned RAM 22 GB/s), load implementation, model and regime; no run measures LMCache restore under concurrency at 16k-64k (04-conflicts.md, conflicts 1 and 4).
- The repo's crossover is calibrated three ways: ~64k measured on the disk path, 32,768 as the connector default, 8,192 in the contention bench (04-conflicts.md, conflict 12).
- Whole-prompt keying gets zero hits on prefix-sharing traffic: `vllm bench serve prefix_repetition`, connector mean TTFT 3355 ms vs 610 ms no-cache vs 183 ms prefix cache (vllm-bench-serve-prefix-repetition).
- The residency cost model ("2-6x cheaper") uses assumed rates (GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr) over HF prefill times; not measured (residency-cost-model). Real proxies now exist: AWS host RAM ~$0.0036/GiB-hr on-demand, local NVMe ~$0.0001/GB-hr, object storage $0.0147-$0.06/GiB-month (05-research-gpu-pricing.md).
- vLLM in-engine measurements never exceeded 16k; >=32k crashed the V1 EngineCore in three configs (vllm-warmstart-prefix-cache).

### Unproven

| claim | experiment that proves or kills it | est. cost |
|---|---|---|
| Restore from a tier that survives the instance sustains 0.5-1.75 GB/s into vLLM paged KV | Move 16.8 GB (128k, 8B) from S3-compatible and remote-NVMe stores into vLLM on A100/H100 via the connector; compare LMCache S3 and vLLM OffloadingConnector object tier | ~20 GPU-hours, 5 days |
| At 64k-128k, kill-and-resume beats re-prefill on TTFT and GPU-seconds | Section 4 benchmark, after chunked-prefill save lands | ~40 GPU-hours, 10 days |
| Spot discount on the chosen class exceeds restore + re-placement + interruption loss per session-hour | 7-day interruption log on one cloud and class; feed measured restore cost and the price sheet into `evals/stateful_cost_model.py` | ~170 spot GPU-hours, 7 days |
| Continuous write-through keeps unsaved bytes under notice x bandwidth at N>=16 | Instrument unsaved bytes at kill time; test at 0 s notice | ~8 GPU-hours, 3 days |
| Chunked (multi-step) prefills can be persisted | Fix `vllm_connector.py:608`; re-run the 16k resume that saved nothing (1.00x) | ~4 GPU-hours, 3 days |
| Portability holds at TP>1 and across attention backends / GPU archs | Cross-instance test at TP=2, mixed backends | ~16 GPU-hours, 5 days |
| Compacted or fp8 state lowers the bandwidth gate | Repeat the kill test with fp8 KV and with compaction | ~24 GPU-hours, 7 days |

## 7. MVP and 6-week build plan

**Ships first:** a hosted, single-cloud, TP=1 endpoint for one 7-8B model with a `session` field on interruptible H100 capacity, plus the published kill-and-resume benchmark. Not in the MVP: TP>1, mid-decode resume, multi-cloud.

- **Week 1: the kill gate.** Measure durable-tier restore bandwidth (Unproven row 1) with the dexa blob format and the LMCache S3 backend as comparator. If no tier clears the derived range at 128k, stop.
- **Week 2: connector.** Chunked-prefill save (`/home/user/dexa/src/dexa/engine/vllm_connector.py:608`); port the block-keyed blake2b prefix chain, single contiguous transfer and staging cap from `/home/user/voice-inference/vkv/backends/voice_kv_connector.py`; keep the dexa connector's async load and adaptive policy.
- **Week 3: harness.** Adapt `/home/user/voice-inference/vkv/{orchestrator,loadgen,metrics}` and `/home/user/dexa/scripts/modal_connector_xinstance.py` into a kill-injection harness; add server-metric scraping and connector provenance to manifests.
- **Week 4: benchmark.** Run the ladder vs baselines (1)-(3); publish wins and losses.
- **Week 5: fleet.** Session registry, preemption handler, spot acquisition on one cloud, interruption telemetry. Reuse the session API and tier calculator in `/home/user/dexa/dexa_platform/sessions/` and the Worker/Durable-Object skeleton in `/home/user/dexa/edge/` (never deployed; wrangler placeholders remain).
- **Week 6: pilot.** Gateway (`/home/user/dexa/dexa_platform/gateway/`, currently forces `stream=False`; needs SSE) with metering (`/home/user/dexa/dexa_platform/control/`); two design-partner agent teams.

**Reused:** dexa connector (917 lines, tested at TP=1), blob/npz format and SessionStore (`/home/user/dexa/src/dexa/session/`), voice connector transfer pattern, voice harness (~2,000 lines; 287,015 measured turns), dexa_platform gateway/control/sessions (29 tests), `evals/stateful_cost_model.py`. **New:** durable-tier backend, session registry and router, preemption handler, fleet manager, streaming, TP>1.

## 8. Pricing model

Two meters. (a) A per-session-hour charge only while a session is running (the shape of Anthropic Managed Agents: $0.08/session-hour, idle not billed) covering durable-tier storage and restore. (b) A per-token rate on an explicit **preemptible tier** priced off spot GPU cost (H100 Spot $2.15-$2.59/GPU-hr on Nebius/AWS versus $3.39-$6.88 on-demand), with a published restore SLO instead of a lost-session risk. The architecture is what makes (b) expressible: for a stateless per-token provider a preemption is a dropped request, so cheap capacity can only be sold by admission gating (Morph's Standby tier bills 50% of standard rates and admits requests only when fleet capacity is under ~25%, returning 429 otherwise). With portable KV the cost of a preemption is bytes divided by bandwidth, bounded and priceable per session. Cache reads can bill at $0 as Tensormesh does, because restore cost sits in the session-hour meter. Whether the class-specific spot delta (19% on A100, 62-81% on H100) leaves margin after restore overhead is Unproven row 3.

## 9. Competitive facts

| who | adjacent thing shipped | not shipped (per research files) | source |
|---|---|---|---|
| LMCache / Tensormesh | S3, NIXL, GDS, disk backends; "persistent KV cache storage"; SaaS with $0 cached input; Operator "coming soon" | no spot/preemption product; no TTL or idle-aware eviction documented | 05-research-kvcache.md, 05-research-agents.md |
| vLLM OffloadingConnector | CPU primary + Filesystem/S3/P2P secondary tiers (developer preview); RFC #38260 TieringManager | no session registry or instance migration | 05-research-kvcache.md |
| llm-d | precise prefix routing; P2P KV pull over NIXL; session-affinity routing | P2P off by default; peers must share block size and hash seed or transfers silently drop to zero | 05-research-kvcache.md |
| NVIDIA Dynamo KVBM | G1-G4 tiers incl. S3-compatible object; TTL retention semantics | no published object-tier performance; disk tier "available, unvalidated" for vLLM | 05-research-kvcache.md, 05-research-agents.md |
| Mooncake Store | DRAM + SSD offload, leases, pins; 92.2% hit rate on Codex traces | DFS tier "not production-ready"; SSD allocator truncates on restart | 05-research-kvcache.md |
| vLLM production-stack | roadmap P0 "request migration on instance failure" | roadmap, not shipped | 05-research-kvcache.md |
| Fireworks | per-replica cache, session affinity, 50% cached discount | cache "only works within 1 replica" | 05-research-agents.md |
| Morph | Standby tier at 50% rates gated on <~25% fleet capacity (429 otherwise); sticky KV placement | no session migration or restore described | 05-research-morph.md |
| Wafer | dedicated endpoints; per-hardware optimization; ZDR | no spot/preemptible tier described | 05-research-wafer.md |
| Anthropic Managed Agents / OpenAI / AgentCore | sessions billed per running hour; 24h cache retention (OpenAI); idle-free microVMs | none sells guaranteed pinned KV; whether Managed Agents keeps model-side cache warm is undocumented | 05-research-agents.md, 05-research-caching.md |

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| Durable-tier restore too slow | GB/s into vLLM paged KV from S3-compatible and remote NVMe at 128k, 8B | < 0.5 GB/s (below the eager-derived floor) | > 1.75 GB/s (above the linear-derived ceiling) |
| Crossover never arrives in-engine | kill-and-resume TTFT vs baseline (1) at 64k and 128k | resume slower than re-prefill at 128k | resume faster with restore GPU-seconds < 20% |
| Spot economics on the chosen class | 7-day interruption log + price sheet into the cost model | discount below restore + re-placement + interruption loss (A100 at 19% off is the stress case) | discount exceeds overhead by a margin the founder sets |
| Notice window | unsaved bytes at kill time under continuous write-through | any session loses turns at 0 s notice | zero lost turns at 0 s notice, N>=16 |
| Concurrency regression | P99 TTFT under 16-24 sessions with restores in flight | P99 > 1.2x baseline (1) (ledger shows 1.6-2.2x today) | within 1.2x |
| TP>1 portability | cross-instance identity at TP=2 | not bit-identical or unbuildable in 2 weeks | identical |
| API drift | connector against vLLM 0.28 | > 1 engineer-week per release | < 2 days per release |

## 11. Founder decisions

- **Model and TP class.** 7-8B / 30B-A3B at TP=1 (validated) vs GLM/Kimi-class at TP=8 (where public agent traffic is; KV 8x replicated). Informed by the TP>1 experiment.
- **GPU class and cloud.** The spot discount is 57-81% on H100/B200 at hyperscalers but 16-19% on L40S/A100 at AWS and zero on Azure H200; neocloud on-demand H100 sits near hyperscaler Spot. Which class and provider, and whether to arbitrage across them, is not judged here. Informed by the interruption log.
- **Whether adjacency matters.** LMCache S3, vLLM OffloadingConnector, llm-d P2P, Dynamo KVBM and production-stack's migration roadmap all touch the mechanism; none ships a spot-session product per the research files. Moat gap or closing window is the founder's call.
- **Hosted vs BYOC vs OSS connector.** Hosted captures the spot margin; BYOC sells connector + control plane into the customer's fleet (compose recipe exists); OSS follows the LMCache-to-Tensormesh path.
- **Which agent segment first.** Coding (128k contexts, human gaps to 20 min) vs computer-use vs back-office. Voice (2.5k contexts, 300 ms p95 gate) is where portable KV adds nothing on speed. Market size not judged.
- **Whether to sell determinism at small context.** At <=16k the only measured advantage is identical continuation.
- **Meter shape.** Per token, per session-hour, or 30% of estimated savings (Tensormesh's post-v1 formula).

## 12. Combinations

- **Stateful session provider (dexa STATEFUL_SESSIONS direction).** Same connector and registry; J adds the durable tier and fleet manager, and the tiering policy that is advisory in `dexa_platform/sessions/tiering.py` becomes enforced because the durable tier is mandatory.
- **Voice session residency (voice-inference arm D).** Same connector engineering (contiguous transfers, staging cap, hint-driven pinning); a 300 ms gate cannot absorb re-placement, so shared code, separate fleet.
- **Compaction / fp8 state.** The dexa repo's own conclusion is that raw-KV resume wins on GPU when the bytes shrink; fp8 halves them losslessly at 48 tokens; compaction is unproven at multi-fact fidelity. Either lowers J's bandwidth gate.
- **Batch document AI.** Stateless, spot-native work that fills the same interruptible fleet between agent turns; the 0.925 DocVQA endpoint in `serve/modal_doc_vlm_serve.py` needs no portable KV.