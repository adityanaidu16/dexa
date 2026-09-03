# J. Preemptible-GPU inference made safe by portable session state

*Run long-lived agent sessions on spot GPUs by making per-session KV portable to a durable tier and restorable on any instance; value is conditional on an unmeasured fast-durable-tier bandwidth at the customer's real context size, with only identical continuation proven below that crossover.*

**Evidence grade C.** The direction splits into sub-claims with different standings. Portability/losslessness is proven in-engine at TP=1 on OPT-125m and Qwen3-0.6B (A10G) and at HF level on Llama-3.1-8B/A100 (identical_output at 8k-64k). The restore-beats-re-prefill claim is contradicted inside the dexa repo: pinned-RAM LMCache restore wins 10-17x single-request at 4k-16k, while the only off-instance tier measured (Modal Volume, ~0.6-0.9 GB/s) loses 0.37x at 8k and 1.6-2.2x on P99 under 16-24-way concurrency, with the ~64k+ crossover never run because chunked-prefill save is missing. Everything that makes J distinct from the existing stateful-session direction (fast durable tier, resume storm behaviour, session registry, preemption handler, fleet manager, spot economics, customer context distribution) is unmeasured or unbuilt. All in-repo numbers are single-seed, TP=1, non-MLA, in-engine capped at 16k, and the concurrent evidence comes from 2.5k-token synthetic voice sessions on a 46 GB L40S with LMCache 0.3.2. Net: measured but bounded/contradicted, leaning D on the product-specific components.

**Weeks to first credible public proof:** 8-10 weeks to a credible public kill-and-resume benchmark at TP=1 on Llama-3.1-8B (three seeds, 8k-128k ladder, baselines vanilla vLLM prefix caching / LMCache 0.5.x S3+NVMe / vLLM OffloadingConnector, per-point tier GB/s). Assumptions: 1-2 engineers full-time; ~$3-6k GPU spend at Modal/spot rates (~150-250 GPU-hours incl. reruns); an S3/remote-NVMe backend for the dexa SessionStore or LMCache's backends lands in week 1 and the tier gate (>=0.5 GB/s per request at 128k) passes; chunked-prefill save (vllm_connector.py:608) plus the port of the voice connector's block-keyed contiguous-transfer path and a rebase from vLLM 0.24.0/0.9.2 to 0.28.0 take 2-3 weeks; generalizing the voice harness (text prompts, aiohttp abort, server-metric scraping, provenance, multi-seed) takes 2 weeks; runs and write-up 2 weeks. The draft's 6-week figure assumed the tier backend, harness generalization and API rebase were free and included a design-partner pilot with partners that do not exist in the research files. Add 4-6 weeks if the proof must include the buyer's TP>1/MLA model class; the hosted spot fleet, registry, preemption handler and streaming gateway are beyond the benchmark and not costed here. The week-1 gate can end the effort at week 1.

---

## 1. Thesis

Candidate J is an inference service that runs long-lived agent sessions on preemptible (spot) GPUs and makes that safe by making every session's KV state portable: written off the GPU continuously, addressed by session, restorable on another instance. The buyer runs open-weight models behind a coding, computer-use or back-office agent. The sentence a customer would repeat: "Our agents run on the cheapest GPUs, and when one gets pulled the session comes back on another one instead of dying."

What the repos establish. (1) Portability and losslessness are proven in-engine at TP=1 on small models and at HF level on Llama-3.1-8B (01-evidence-ledger-dexa.md, cross-instance-resume). (2) Restoring from pinned host RAM beats re-prefill inside vLLM+LMCache at 4k-16k, single request only (lmcache-offload-restore). (3) Restoring from the one off-instance tier measured (a Modal Volume, ~0.6-0.9 GB/s) loses to vLLM re-prefill at 8k (0.37x) and under 16-24-way concurrency (P99 1.6-2.2x worse); the repo's crossover estimate for that path is ~64k+ (04-conflicts.md, conflict 1). So the thesis is conditional: portable KV adds something a transcript replay on instance B does not only if the durable tier is fast enough at the customer's actual context size; below that crossover the only measured advantage is identical continuation. The two deciding numbers, fast-durable-tier restore bandwidth into vLLM paged KV and the customer's real context distribution, are unmeasured.

## 2. Customer and workload

**Who buys.** Infrastructure engineers shipping agents on open weights: coding-agent products (OpenHands recommends Qwen3.6-35B-A3B on vLLM/SGLang; Cursor's Composer 2 started from Kimi K2.5), computer-use products (Browser Use ships a 30B-A3B open model), enterprises self-hosting a harness. Open-weight models carry a majority of OpenRouter tokens by mid-2026, concentrated in coding and agentic workloads (05-research-agents.md); no survey of how many agent teams self-host exists.

**What they run.** vLLM (0.28.0) or SGLang (0.5.18) behind an OpenAI-compatible endpoint; models from ~30B-A3B MoE up to GLM-5.2/Kimi-K3-class MoE at TP=8, where "TP=8 keeps eight copies of the KV cache" and the KV is MLA (llm-d). Context, from frontier-model traces: TraceLab's ~4,300 Claude Code/Codex sessions show a median prefix of 126,180 (Claude) / 115,584 (Codex) tokens per step, ~860 new input, ~200-250 output; llm-d's 219 production Claude Code sessions show a median request of 195K input / 317 output, 96% of requests reusing >=90% of input verbatim. Caveat the draft omitted: these are Claude/Codex traces; the only open-weight context guidance found is OpenHands recommending 22k minimum / 32k for its local model, below the repo's measured crossover. Idle pattern: human gap median 1.4 min, p90 20.6 min; per-request time median 38.3 s (TraceLab); tool gaps 5.2 s median / 81.4 s P99 (vLLM/Mooncake). A session is busy roughly every 40 s inside a turn and idle between human turns. Turn length median ~45 s, 99.9th percentile over 45 min (Anthropic). Fleet concurrency and context distribution per buyer: unmeasured.

**How they pay today.** Hosted dedicated endpoints (Fireworks H100 $8.00/GPU-hr from Sep 1 2026; Baseten $6.50; Together $3.99) or on-demand GPU-hours. Spot, per 05-research-gpu-pricing.md: AWS p5 H100 $6.88 on-demand vs $2.59 Spot (~62% off); p6-b200 $14.24 vs $5.24 (~63%); Azure ND H100 $12.29 vs $2.37 (~81%); Nebius H100 $3.85 vs $2.15. Class-specific: AWS A100 p4de Spot ~19% off, L40S g6e ~16%, Azure H200 Spot equal to list. Relative to neocloud on-demand (H100 median $3.39; Vast $1.73), hyperscaler Spot is roughly at, not below, cheap on-demand. Notice: AWS 2 min best-effort; GCP 0 s default or 120 s preview; Azure 30 s best-effort; Vast none. GPU-pool interruption rates: not retrievable (AWS publishes <5% to >20% bands, Azure per-hour rates).

## 3. The pain, in the customer's words

- "Spot is 60-80% off list on H100/B200, but I can't put a 40-minute agent turn on a box with a 30-second or zero-second notice." (05-research-gpu-pricing.md; Anthropic 99.9th-percentile turn >45 min.)
- "When a replica dies or traffic overflows, we re-prefill 130k tokens and the user waits." (OpenAI docs: overflow is a cache miss; Fireworks: cache "only works within 1 replica".)
- "Past ~150K tokens our Codex runs on Azure got cached_input_tokens = 0; 3.28M re-billed tokens, 71% of fresh input, in one 4-turn run." (openai/codex #25604.)
- "The provider moved us from a 1-hour to a 5-minute cache and the bill went up 17%." (anthropics/claude-code #46829: $949 over 119,866 calls.)
- "Nobody sells a session guaranteed to stay warm." (05-research-agents.md: no provider found selling guaranteed session-pinned KV retention; closest are DeepInfra's paid 5m/1h retention and Fireworks/Baseten affinity routing.)

## 4. Value proposition and the proof-of-value benchmark

**Value proposition.** For sessions above the restore/re-prefill crossover, a portable KV tier lets the session live on interruptible capacity: a preemption becomes a restore on another instance, the same path as an idle-eviction restore. Below the crossover the product adds identical continuation, not speed or cost.

**The inequality.** Restore beats re-prefill iff (tier bandwidth available to this request) x (vLLM re-prefill time for this context) > (KV bytes) (04-conflicts.md, conflict 1). The ~64k+ disk-path crossover was never run because chunked-prefill persistence is missing (chunked-prefill-gap). Derived requirement, arithmetic not measurement: Llama-3.1-8B KV is 131 KB/token, so 128k is ~16.8 GB (2x the recorded 8.4 GB at 64k); vLLM prefills 8k in 617 ms and 128k in 32.68 s (eager, prefix caching off); the tier therefore needs ~0.5 GB/s (eager floor) to ~1.7 GB/s (linear from 8k), ~0.8 GB/s at 64k. External points, none reproduced: LMCache/CoreWeave report 8.5-10 GB/s per 8-GPU node from object storage; the LMCache paper states remote-storage loading only beats prefill above ~256K tokens at 32 Gbps in its setup; VAST/Backend.AI report NVMe/GDS restore at ~140K contexts cutting average TTFT 2.09x on a 128B model. Neither connector saves decode-token KV, so the MVP resumes at step boundaries.

**The benchmark: kill-and-resume.** N concurrent agent sessions replayed from a trace; instance A is SIGKILLed with no drain; the sessions' next steps route to instance B. Measured: (a) resume TTFT on B, p50/p95; (b) GPU-seconds (SM-busy) consumed on B to resume; (c) `identical_output` versus an unbroken run, chunked prefill on both sides; (d) aggregate GB/s into B during the resume storm; (e) sessions sustained per GPU at a p95 gate. Ladder 8k/32k/64k/128k; Llama-3.1-8B (native 128k) at TP=1 first, the only validated regime; three seeds.

**Baselines, named.** (1) Vanilla vLLM with prefix caching, re-prefilling from the message log on B: the buyer's own fallback and the baseline that beat the connector at 8k. (2) vLLM + LMCache 0.5.4 S3-compatible and NVMe backends. (3) vLLM's native OffloadingConnector with an object-store secondary tier. (4) llm-d P2P pull from a surviving peer's CPU tier (published crossover ~8K-12K tokens; 48K-token pull 235 ms vs 1,988 ms recompute).

**Pre-registered target.** At L=128k: resume p95 TTFT on B below baseline (1); restore GPU-seconds below 20% of re-prefill GPU-seconds; `identical_output` true; p95 TTFT at N>=16 no worse than baseline (1) by more than 1.2x; N=16 simultaneous restores complete within 2x the single-session restore time. At L<=16k we publish the loss.

**Why a skeptical buyer believes it.** A kill test, not a warm-cache test; baseline (1) is their own fallback; identity is checked; losses are published next to wins; per-point tier GB/s lets the buyer plug in their own storage.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| vLLM engine | OSS, unmodified | prefill/decode, paged KV, prefix caching |
| Session KV connector | custom (KVConnectorBase_V1), or LMCache/OffloadingConnector (founder decision) | continuous write-through of block-keyed KV; async contiguous restore; chunked-prefill save |
| Durable KV tier | OSS stores (S3-compatible / remote NVMe / GDS) + blob format | holds session KV off-instance; the tier that survives preemption |
| Session registry + router | custom | session id -> KV location and instance; re-placement on kill |
| Preemption handler | custom | on notice (where one exists): flush unsaved blocks, drain, re-route |
| Spot fleet manager | custom | acquires/releases interruptible capacity; price and interruption telemetry |
| Gateway / control plane | custom (dexa_platform, edge) | OpenAI-compatible API with a `session` field, keys, metering |
| Load harness | custom (voice-inference vkv) | trace replay, kill injection, metrics |

Differentiation lives in the connector's save/load path and the session control plane, not in the model, kernels or scheduler; vLLM is built on, not replaced. Because GCP's default notice is 0 s, the connector writes through continuously and the notice window covers only unsaved blocks. Standing facts: KVConnector V1 has been labeled experimental since its April 2025 merge; the in-repo connectors target vLLM 0.24.0 (dexa) and 0.9.2 (voice) while current is 0.28.0; both assume a non-MLA paged layout and TP=1, so the buyer's MLA/TP=8 models are outside anything validated.

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
   | connector: async restore (N sessions at once = resume storm)
 vLLM on spot instance B  <-+   (identity checked)
```

## 6. Evidence

### Proven

- KV saved by one vLLM process loads in a separate process with prefill skipped and identical output: two Modal containers (pids 36 vs 216) sharing a Volume, `identical_output=True` on OPT-125m/A10G; 8176/8192 tokens matched block-aligned on Qwen3-0.6B/A10G across three containers (cross-instance-resume). TP=1, single attention backend, single-step prefill, prefix caching off; 10/10 hook signatures match vLLM 0.24.0 (connector-conformance).
- Lossless persist/resume at 8k-64k on Llama-3.1-8B/A100 via the HF persist bench: `identical_output` true at every length; state 1.0/2.1/4.2/8.4 GB. HF-level, not in-engine; its 3.7-5.5x speedups are against HF eager prefill and superseded for the vLLM target (04-conflicts.md).
- Fast-tier restore beats re-prefill in real vLLM at small context: LMCache CPU offload/restore 301->30 ms at 4k (10.0x), 1,320->77 ms at 16k (17.1x); 16,384 tokens in 39.8 ms at 22 GB/s. Qwen2.5-7B, single request, prefix caching off, LMCache unpinned. HF tensor-copy restore 11.8x at 4k to 28.9x at 64k and NVMe (naive torch.load) 1.2x-7.0x, vs an HF baseline 5-7x slower than vLLM.
- fp8/int8 KV at 0.5x bytes: 100% greedy agreement over 48 tokens; int4 diverges at token 14 (one passage, HF). KV is keyed to exact weights (base<->instruct diverges within 2-5 tokens).

### Bounded / contradicted

- Raw-KV load through the Dexa connector loses to vLLM re-prefill: 8B/A100/8k resume 1681 ms vs 617 ms (0.37x); 24x3072 async P99 8930 vs 5596 ms; 16x8192 P99 20296 vs 9379 ms; the adaptive policy only matches baseline (5603/9371 ms). Contradicts the 10-34x entries on tier (Modal Volume vs pinned RAM), load path, model and regime; no run measures a RAM- or NVMe-tier restore under concurrency at 16k-64k (conflicts 1, 4). Single seed; ~1.7x box-to-box Modal variance.
- Crossover calibrated three ways: ~64k measured, 32,768 connector default (`vllm_connector.py:392`), 8,192 in the contention bench (conflict 12).
- Whole-prompt keying gets zero hits on prefix-sharing traffic: connector mean TTFT 3355 ms vs 610 ms no-cache vs 183 ms prefix cache (OPT-125m/A10G); saves are unconditional.
- Voice connector under load: a contiguous-transfer connector held N=300 on an L40S (whole-run p95 316.8 ms; full-load-window 336.8 ms) where LMCache 0.3.2's tested configs collapsed (25,997.7 ms default; 6,025.1 ms layerwise); it FAILED the 300 ms gate, max passing N unchanged at 200; single seed; ~2.5k-token sessions; restore latency never timed; server-side counts unarchived (02-evidence-ledger-voice.md). Staging buffers outside vLLM's memory budget OOM'd the engine at 345 concurrent requests (prose only).
- Bit-identical continuation is proven only for same-backend single-step prefill; the voice correctness suite refuses text match because greedy decoding "cascades cached-vs-full-prefill kernel numerics into divergent text" (vi-kv-restore-correctness). Identity across chunked prefill, backends, TP and GPU archs is unmeasured.
- Residency cost model ("2-6x cheaper") uses assumed rates (GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr) over HF prefill times; not measured (real proxies: host RAM ~$0.0036/GiB-hr, object storage $0.0147-$0.06/GiB-month).
- In-engine measurements never exceeded 16k; >=32k crashed the V1 EngineCore in three configs (Qwen2.5-7B, YaRN).

### Unproven

| claim | experiment that proves or kills it | est. cost |
|---|---|---|
| A tier that survives the instance sustains 0.5-1.7 GB/s per request into vLLM paged KV | Move 16.8 GB (128k, 8B) from S3-compatible, remote-NVMe and GDS stores into vLLM on A100/H100; compare LMCache 0.5.x and OffloadingConnector | ~20 GPU-hours, 5 days |
| At 64k-128k, kill-and-resume beats re-prefill on TTFT and GPU-seconds | Section 4 benchmark, after chunked-prefill save lands | ~40 GPU-hours, 10 days |
| Resume storm: N>=16 sessions restoring into one instance stay within 2x single-session time | Kill instance A holding N sessions; log aggregate GB/s and p95 resume on B | ~10 GPU-hours, 3 days |
| Customers' real open-weight agent contexts sit above the crossover | Context-length histogram from two design partners' gateway logs | 0 GPU-hours, 2 weeks of access |
| Spot discount on the chosen class exceeds restore + re-placement + interruption loss | 7-day interruption log on one cloud and class; measured restore cost and price sheet into `evals/stateful_cost_model.py` | ~170 spot GPU-hours, 7 days |
| Chunked (multi-step) prefills can be persisted | Fix `vllm_connector.py:608`; re-run the 16k resume that saved nothing (1.00x) | ~4 GPU-hours, 3 days |
| Identity holds across chunked prefill, TP>1 and MLA layouts | Cross-instance test at TP=2, chunked prefill on, one MLA model | ~16 GPU-hours, 5 days |

## 7. MVP and 6-week build plan

**Ships first:** the published kill-and-resume benchmark on one 7-8B model at TP=1, plus a hosted single-cloud endpoint with a `session` field on interruptible capacity. Not in the MVP: TP>1, MLA, mid-step resume, multi-cloud, design-partner pilot (past week 6; no design partners exist in the research files).

- **Week 1: the tier gate.** Write an S3-compatible and remote-NVMe backend for `SessionStore` (`/home/user/dexa/src/dexa/session/store.py`, today a local directory) or drive LMCache 0.5.x's S3/GDS backends; measure GB/s into vLLM at 128k on Llama-3.1-8B. If no tier clears ~0.5 GB/s per request, the direction is determinism-only; stop or re-scope.
- **Weeks 2-3: connector.** Chunked-prefill save (`/home/user/dexa/src/dexa/engine/vllm_connector.py:608`); port the block-keyed blake2b prefix chain, single contiguous transfer and 1 GB staging cap from `/home/user/voice-inference/vkv/backends/voice_kv_connector.py` (vLLM 0.9.2) onto the dexa connector (0.24.0) and rebase on 0.28.0. Two weeks because the connectors sit two API generations apart.
- **Week 4: harness.** Generalize `/home/user/voice-inference/vkv/{orchestrator,loadgen,metrics}` and `/home/user/dexa/scripts/modal_connector_xinstance.py` into a kill-injection harness: text prompts, aiohttp abort, server-metric scraping, connector provenance, multi-seed aggregation (all missing per 03-build-inventory.md).
- **Week 5: benchmark.** Run the ladder vs baselines (1)-(3), three seeds; publish wins and losses with per-point GB/s.
- **Week 6: fleet skeleton.** Session registry, preemption handler, spot acquisition on one cloud, interruption telemetry. Reuse the session API in `/home/user/dexa/dexa_platform/sessions/` (tiering advisory; warm inferred from turn count) and the Durable-Object skeleton in `/home/user/dexa/edge/` (never deployed). The gateway (`/home/user/dexa/dexa_platform/gateway/app.py:127` forces `stream=False`) needs SSE before any pilot.

**Reused:** dexa connector (917 lines, TP=1), blob/npz format and SessionStore, voice connector (574 lines), voice harness (~2,000 lines; 287,015 measured turns), dexa_platform (29 tests; no integration test against a live backend), `evals/stateful_cost_model.py`. **New:** durable-tier backend, session registry and router, preemption handler, fleet manager, streaming, TP>1/MLA.

## 8. Pricing model

Two meters. (a) A per-session-hour charge only while a session is running (the shape of Anthropic Managed Agents: $0.08/session-hour, idle not billed) covering durable-tier storage and restore. (b) A per-token rate on an explicit **preemptible tier** priced off spot GPU cost (H100 Spot $2.15-$2.59/GPU-hr versus hosted dedicated H100 at $3.99-$8.00 and $3.39 median neocloud on-demand), with a published restore SLO. The architecture makes (b) expressible: for a stateless per-token provider a preemption is a dropped request, so cheap capacity is sold by admission gating (Morph's Standby tier: 50% of standard rates, admitted only under ~25% fleet capacity, 429 otherwise). With portable KV a preemption costs bytes divided by bandwidth, priceable per session. Cache reads can bill at $0 as Tensormesh does. Whether the class-specific spot delta leaves margin after restore overhead is Unproven row 5; whether the customer's reference price is neocloud on-demand (delta roughly zero) or hosted dedicated (~50-70%) is a founder decision.

## 9. Competitive facts

| who | adjacent thing shipped | not shipped (per research files) | source |
|---|---|---|---|
| LMCache 0.5.4 / Tensormesh | CPU, disk, S3, NIXL, GDS backends; SaaS with $0 cached input; Operator "coming soon" | no spot/preemption product; no TTL or idle-aware eviction documented | 05-research-kvcache.md |
| vLLM OffloadingConnector / production-stack | CPU primary + Filesystem/S3/P2P secondary tiers (developer preview); RFC #38260 TieringManager; roadmap P0 "request migration on instance failure" | no session registry or instance migration shipped | 05-research-kvcache.md |
| llm-d v0.9 | precise prefix routing; P2P KV pull over NIXL; session-affinity routing | P2P off by default; peers must share block size and hash seed or transfers silently drop to zero | 05-research-kvcache.md |
| NVIDIA Dynamo KVBM / Mooncake | G1-G4 tiers incl. S3, TTL retention semantics (KVBM); DRAM + SSD offload, leases, pins (Mooncake) | no published object-tier performance; disk tier "available, unvalidated" for vLLM; Mooncake DFS "not production-ready" | 05-research-kvcache.md |
| SGLang HiCache | L1/L2/L3 with Mooncake/3FS/NIXL/file backends; open RFCs for pin/demote hints with bounded TTL | L1/L2 private per instance; RFCs open, not merged | 05-research-kvcache.md |
| Fireworks | per-replica cache, x-session-affinity, 50% cached discount | cache "only works within 1 replica"; no preemptible tier | 05-research-providers.md |
| DeepInfra / Baseten | paid explicit 5m/1h retention on two models (DeepInfra); KV-aware routing, CPU offload, disk cache (Baseten) | no session migration or spot tier described | 05-research-providers.md, 05-research-agents.md |
| Morph / Wafer | Standby tier at 50% rates gated on <~25% fleet capacity, sticky KV placement (Morph); dedicated endpoints, per-hardware optimization, ZDR (Wafer) | no session migration, restore, or preemptible tier described | 05-research-morph.md, 05-research-wafer.md |
| Anthropic Managed Agents / OpenAI / AgentCore | sessions billed per running hour; 24h cache retention (OpenAI); idle-free microVMs | none sells guaranteed pinned KV | 05-research-agents.md, 05-research-caching.md |

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| Durable-tier restore too slow | GB/s per request into vLLM paged KV from S3-compatible, remote NVMe and GDS at 128k, 8B | < 0.5 GB/s (eager-derived floor) | > 1.7 GB/s (linear-derived ceiling) |
| Crossover never arrives in-engine | kill-and-resume p95 TTFT vs baseline (1) at 64k and 128k | resume slower at 128k | faster, with restore GPU-seconds < 20% |
| Resume storm | N=16 sessions restoring into one instance after a kill | p95 resume > 2x single-session restore | within 2x |
| Customer context below crossover | context histogram from two design partners | median step context < measured crossover | median above it |
| Identity across kernel paths | `identical_output` with chunked prefill on, then TP=2, then one MLA model | not identical on the TP=1 chunked path | identical on all three |
| Spot economics on the chosen class | 7-day interruption log + price sheet into the cost model | discount below restore + re-placement + interruption loss (A100 at 19% off is the stress case) | exceeds overhead by a founder-set margin |
| Notice window | unsaved bytes at kill under continuous write-through | any completed step lost at 0 s notice | zero lost steps at 0 s notice, N>=16 |
| Concurrency regression | P99 TTFT under 16-24 sessions with restores in flight | P99 > 1.2x baseline (1) (ledger: 1.6-2.2x, disk tier) | within 1.2x |
| API drift | connector rebased from 0.24.0/0.9.2 to 0.28.0 | > 1 engineer-week per release | < 2 days per release |
| Write-through throughput cost | req/s with saves on vs baseline (1) | < 0.8x (ledger: 9.8 vs 52.3 req/s, whole-prompt keying) | > 0.9x |

## 11. Founder decisions

- **Whether continuity-only is a product.** If the tier gate fails, J is a session-affinity router with transcript replay plus deterministic continuation. Sell that, wait for fp8/compaction to lower the gate, or stop. Informed by Week 1.
- **Reference price for the spot delta.** Hyperscaler list (57-81% off), neocloud on-demand (roughly zero), or hosted dedicated ($3.99-$8.00 H100). Which cloud and class; informed by the interruption log.
- **Model and TP class for the proof.** 7-8B at TP=1 (validated) vs 30B-A3B vs GLM/Kimi-class MLA at TP=8 (where public agent traffic is; KV 8x replicated). The MVP proves nothing about the latter.
- **Build the connector or build on one.** Custom (voice: contiguous transfers held N=300 where LMCache 0.3.2 collapsed) vs LMCache 0.5.x (S3/GDS/NIXL backends exist; 22 GB/s single-request) vs vLLM OffloadingConnector (object tier in preview). Informed by Week 1 running both.
- **vLLM vs SGLang.** The repos validated only vLLM; SGLang HiCache has L3 backends and open session-hint RFCs. Not judged here.
- **Whether adjacency matters.** LMCache S3, OffloadingConnector, llm-d P2P, Dynamo KVBM and storage-vendor tiers all touch the mechanism; none ships a spot-session product per the research files. Moat gap or closing window is the founder's call.
- **Hosted vs BYOC vs OSS connector.** Hosted captures the spot margin; BYOC sells connector + control plane into the customer's fleet; OSS follows the LMCache-to-Tensormesh path.
- **Which agent segment first.** Coding (frontier traces 115-195k; open-weight guidance 32k) vs computer-use vs back-office; voice (2.5k contexts, 300 ms gate) is where portable KV adds nothing on speed. Market size not judged.
- **Whether to sell determinism at small context.** Proven only for same-backend single-step prefill; voice records divergence across kernel paths.
- **Meter shape.** Per token, per session-hour, or 30% of estimated savings (Tensormesh's post-v1 formula).

## 12. Combinations

- **Stateful session provider (dexa STATEFUL_SESSIONS direction).** Same connector and registry; J adds the durable tier and fleet manager, and the advisory policy in `dexa_platform/sessions/tiering.py` becomes enforced.
- **Voice session residency (voice-inference arm D).** Same connector engineering (contiguous transfers, staging cap, hint-driven pinning at 1.5x, single seed); a 300 ms gate cannot absorb re-placement: shared code, separate fleet.
- **Compaction / fp8 state.** fp8 halves bytes losslessly over 48 tokens; compaction loses multi-fact fidelity at 128x (AM 0.53 vs H2O 0.82). Either lowers J's bandwidth gate; neither is measured in-engine (~24 GPU-hours to test fp8).
- **Batch document AI.** Stateless, spot-native work filling the same interruptible fleet between agent turns; the 0.925 DocVQA endpoint in `serve/modal_doc_vlm_serve.py` (200 pages, relaxed match) needs no portable KV.
