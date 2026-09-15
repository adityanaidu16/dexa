# F. Portable, persistent KV state layer for self-hosters (open-core infrastructure software)

*An Apache-2.0 vLLM/SGLang connector plus durable, tenant-keyed KV store that lets an agent session survive pod death and resume on any replica with the same output, sold with a commercial fleet control plane; the portability core is measured once in-engine at TP=1, the speed story is contradicted, and durability/governance are unbuilt.*

**Evidence grade C.** The one in-engine measurement of the core claim (cross-instance resume, bit-identical output, prefill skipped) is real but thin: OPT-125m with two external blocks on vLLM 0.24.0, plus a block-aligned 8k resume on Qwen3-0.6B that is only next-token-correct; the 8B lossless result is HF-level (HFBackend), not the serving engine. Every speed number is contradicted (0.37x single-request, 1.6-2.2x worse P99 under load for the repo's own connector vs 10-17x for LMCache single-request), the only concurrent result is single-seed synthetic voice traffic on vLLM 0.9.2 with a different connector, no pod-kill test exists, and the claimed differentiators (durable tiers, decode-KV save, tenant keying, audit, TP>1, MLA support) are unbuilt. Measured but bounded and contradicted is C.

**Weeks to first credible public proof:** ~10 weeks to the first credible public proof at TP=1 (a Preemption Survival Score run on vllm bench serve + Mooncake traces vs vanilla, OffloadingConnector+S3 and LMCache 0.5.x layerwise, Llama-3.1-8B on A100 and H100, with KV-byte identity and greedy-divergence reported), assuming two senior engineers full-time, ~150-200 GPU-hours, no vLLM fork, and that the vLLM 0.28 port of the two pinned connectors (0.24.0 and 0.9.2) plus chunked-prefill/decode-block save and layer-pipelined async restore each take about a week. The draft's 6 weeks assumed TP>1 in one week against its own 10-day/60-multi-GPU-hour estimate; a TP=2/4 point adds 3-4 weeks and an MLA/hybrid model adds an unestimated amount.

---

## 1. Thesis

Candidate F is an Apache-2.0 infrastructure layer that turns an inference engine's KV cache into a durable, portable, governed object owned by the operator: session state that survives a pod restart, replica move or spot preemption and resumes on a different vLLM process with the same greedy output and no re-prefill beyond block alignment; keyed to exact weights and tenant; retained and evicted by policy rather than the engine's LRU. It is sold to platform teams, neoclouds and regulated deployments that run vLLM or SGLang themselves, with a commercial fleet control plane (placement, tenant policy, audit, quotas) as the paid layer. The sentence a customer would repeat: "When our agent pod dies, the session comes back on another node with the same output and zero recompute, and we can prove which tenant's context was touched."

The measured foundation is narrow: cross-instance resume is PROVEN at TP=1 on a real vLLM 0.24.0, but on OPT-125m with two external blocks, plus a block-aligned 8k resume on Qwen3-0.6B that is next-token-correct; the 8B lossless result is HF-level. Two GPU-validated KVConnector implementations exist (dexa, vLLM 0.24.0; voice, vLLM 0.9.2), neither with all the properties this product needs. No pod-kill test has been run; the two containers in the cross-instance test shared a Modal Volume. The speed story is not the product: the dexa ledger shows raw-KV loading losing to vLLM re-prefill single-request (0.37x at 8B/8k) and under 16-24-way concurrency (P99 1.6-2.2x worse, async). Durability, portability and governance are the thesis; speed parity is a gate.

## 2. Customer and workload

Who buys: the ML-platform team that already operates vLLM (PyPI 0.28.0, Aug 26 2026) or SGLang (0.5.18) on Kubernetes, often via llm-d (CNCF Sandbox, v0.9.0 Aug 17 2026); neoclouds selling dedicated capacity (Tensormesh resells reserved H200s at $2.50/hr via Nebius and Yotta; Wafer cites MI355X rental at $2.29-2.95/GPU-hr); regulated deployments that keep inference state inside their perimeter.

What they run: long-horizon coding and computer-use agents on open-weight models. TraceLab's ~4,300 Claude Code/Codex sessions: median cached prefix 115,584-126,180 tokens per step, ~857-886 new input tokens, 184-252 output tokens; human gap median 1.4 min, p90 20.6 min; prefix-cache hits 95.7% overall, 84.4% on user-initiated steps. llm-d's 219 production Claude Code sessions on GLM-5.2: median 195K input / 317 output tokens, 96% of requests reuse >=90% of input verbatim, inter-turn pauses median 2 s, p99 11.4 min, and "TP=8 keeps eight copies of the KV cache". Codex traces replayed by vLLM/Mooncake: ~131:1 input:output, inter-turn delays 5.2 s median to 81.4 s P99. Alibaba's production traces: ideal hit ratios 54-62%, P99 KV lifespan 97 s in to-B traffic.

Workload caveat the draft omitted: the models these customers run (GLM-5.2, Kimi K2.7/K3, DeepSeek V4, Qwen3.6-35B-A3B; open-weight models carry a majority of OpenRouter tokens by mid-2026, concentrated in coding/agentic) are MoE with MLA or hybrid attention at TP>1. Every repo measurement is dense GQA at TP=1: Llama-3.1-8B and Qwen2.5-7B on A100-80GB, plus OPT-125m and Qwen3-0.6B on A10G and Qwen2.5-7B on L40S; in-engine evidence stops at 16k (vLLM V1 crashed at >=32k with Qwen2.5-7B+YaRN; 32k-64k points are HF-only). Both connectors assume a non-MLA fp16/bf16 paged layout and a single attention backend.

How they pay: GPU-hours, not tokens; re-prefill shows up as prefill throughput (arm A collapse on an L40S: 9,466-10,779 prompt tok/s while the prefix-cache hit rate fell from ~92% to 1.6-1.9%). Hosted comparators: Anthropic 5-minute default TTL (1 hour at 2x write price), OpenAI 30-minute (GPT-5.6+) or up-to-24h retention, DeepSeek disk cache cleared "within a few hours to a few days".

## 3. The pain, in the customer's words

No customer interviews exist in either repo. The statements below are public issues and docs, and most describe hosted-API TTL pain, not a self-hoster losing KV to a pod kill; no self-hoster quote about preemption loss was found in the research files.

- "After a gap longer than the TTL, the next request recomputes the full input and re-establishes the cache" (Claude Code docs). A Codex user on Azure measured 3.28M re-billed tokens, 71% of fresh input, once a session passed ~150K tokens (issue #25604, medium confidence).
- "Host memory cannot be pooled across instances or across hosts, not even for two instances on the same node" (SGLang HiCache best practices).
- llm-d P2P sharing: peers must share identical block size and hash seed or transfers silently drop to zero (paraphrased from the llm-d post).
- "This API is experimental and subject to change" (vLLM KVConnectorBase_V1 docstring, unchanged since the April 2025 merge).
- Building the equivalent in-house "takes 20 engineers and three or four months" (Tensormesh CEO, TechCrunch).
- From the repos: "Memory your engine doesn't know about will kill it" (voice-inference FINDINGS), after connector staging buffers outside vLLM's budget, jointly with unbounded warmup requests, OOM'd the engine at 345 concurrent requests.

## 4. Value proposition and the proof-of-value benchmark

The proof is a correctness-and-durability benchmark, not a TTFT race. 04-conflicts reconciles every restore number in both repos with one arithmetic rule (derived, not measured): restore wins only when the tier's available bandwidth times the engine's prefill time exceeds the KV bytes. In-repo, pinned RAM at ~10-20 GB/s already wins at 4k single-request (LMCache 10.0x); the connector's ~0.6 GB/s Volume path crosses over only near 64k. The honest claim is "same output, zero recompute, on any replica", with speed parity as a gate.

Metric: the Preemption Survival Score.

Setup: two vLLM instances (same weights hash) behind round-robin routing that deliberately defeats local prefix caching (BENCHMARK_PLAN gotcha #1). Drive `vllm bench serve` multi-turn with `--max-active-conversations` forcing eviction and retrieval, plus the Mooncake `conversation_trace` replayed via `--dataset-name timed_trace --self-timed`. Mid-run, SIGKILL instance A after turn k and reschedule its sessions on B. Model: Llama-3.1-8B (native 128k; avoids the YaRN crash) on A100 and H100, TP=1 first.

Three numbers per configuration: (1) correctness: fraction of resumed sessions whose restored KV blocks are byte-identical to the originals AND whose greedy continuation matches an uninterrupted run, reported alongside a control (vanilla in-GPU prefix-cache hit vs cold prefill), because the voice repo's own correctness suite states greedy text can diverge across cached-vs-full-prefill kernel paths even with correct KV; (2) tokens re-prefilled per resume, including decode tokens (block-alignment floor; the ledger measured 8,176 of 8,192 matched, 16 re-prefilled, prompt-only); (3) P50/P99 TTFT after resume versus cold re-prefill at 8k, 32k, 64k and 128k, at concurrency 1, 8, 32.

Baselines, named and configured to survive: vanilla vLLM prefix caching (state dies with the pod); vLLM OffloadingConnector with CPU primary and FS/S3 secondary tier (LRU/ARC; offload_prompt_only defaults true, so decode blocks are skipped unless changed); LMCache 0.5.4 layerwise-on with an S3/Redis/Mooncake L2 and with local CPU only; Mooncake Store with SSD offload (v0.3.13 recovers SSD offload after master restart); llm-d P2P over NIXL (off by default); SGLang HiCache with a Mooncake L3 once an SGLang backend exists. Several of these persist KV off-pod, so survival alone will not separate the product; the separation must appear in re-prefilled tokens (block floor vs 256-token chunks), correctness, tenant keying and P99 under load.

Targets, and why a skeptic would believe them: KV-byte identity on 100% of resumed sessions and greedy identity at least equal to the vanilla control (held on OPT-125m, 2 blocks; next-token-correct at 8k on Qwen3-0.6B and Llama-3.1-8B; TP>1 unproven). Resume TTFT at 128k at least at parity with re-prefill. External anchors: llm-d measured a 48K-token cross-node CPU-to-CPU pull at 235 ms vs 1,988 ms recompute on gpt-oss-120b (8.5x, derived) and cites a ~8K-12K crossover for P2P on GLM-5.2-FP8; the LMCache paper says remote storage beats prefill only above ~256K tokens at 32 Gbps. The ledger's 0.37x came from ~0.6 GB/s Volume loads versus LMCache's self-reported 22.0 GB/s pinned-CPU retrieval (single request, LMCache version unpinned), so the first job is the tier, not the thesis. The skeptic believes it because the harness is not ours, correctness is bitwise, and the repo already published its losing numbers.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| vLLM (target 0.28) / SGLang | OSS, unmodified | Serves tokens; owns HBM; exposes KVConnectorBase_V1 / HiCacheStorage |
| State connector (target; merges `src/dexa/engine/vllm_connector.py`, 917 lines, vLLM 0.24.0, whole-prompt keying, async load, no chunked prefill, and `vkv/backends/voice_kv_connector.py`, 574 lines, vLLM 0.9.2, blake2b block-chain keying, pin-by-hint, synchronous loads, no decode KV) | Custom | Block-hash keyed lookup, chunked-prefill and decode-block save, layer-pipelined async load, pin via `request_finished`. No existing connector has all of these. |
| State object format (`src/dexa/session/blob.py`, DEXAKV01) | Custom | bf16-native, mmap-loadable. Header today: spec, dtype, shapes, positions, token_ids, meta. Weights hash, TP layout, attention backend and tenant are Week 2 additions; the store key today has no tenant component. |
| Pinned-CPU slab + NVMe + object tiers | Custom policy over OSS storage (S3, NVMe; optionally NIXL/Mooncake transport) | Durable copy survives pod death; placement by policy, not engine LRU. Unbuilt beyond the CPU slab. |
| Governance: tenant namespace, TTL/retention, audit log, recompute-avoided metrics | Custom; nothing exists (grep for rbac/audit/prometheus returns only comments) | The v1 differentiator for regulated buyers (Mooncake and Dynamo document tenant isolation; audit of state operations is not documented anywhere in the research files) |
| Segment DAG, incremental recompute, RoPE re-phase (`src/dexa/segment`, 643 lines) | Custom, HF-backend only | Later substrate for edit-in-place sessions; not in the vLLM path |
| Benchmark harness (`vkv/*` ~2,000 lines; `benchmarks/vllm_connector_check.py`; `scripts/modal_bench_contention.py`) | Custom | Independent-harness replays, conformance tiers, contention runs |
| Fleet control plane (commercial) | Custom; `dexa_platform/control` (620 lines) is a keys/metering scaffold | Placement, KV migration, quotas, dashboards |

Where the differentiation lives: not in kernels, the engine or the transport (all OSS). It lives in keying and save semantics (block-hash, decode blocks, chunked prefill), the state-object format (portable across processes and, once built, across TP layouts), the durability policy, and the control plane (who may read which state, for how long, with what audit). vLLM and SGLang are used, not replaced; the connector seam is the one LMCache, NIXL, Mooncake and FlexKV also use, and it is labeled experimental.

```
 agent harness ── OpenAI-compatible API ──► vLLM / SGLang (unchanged)
                                               │ KVConnector / HiCacheStorage
                                               ▼
        ┌───────────── state layer (Apache-2.0) ──────────────────┐
        │ block-hash index (blake2b chain; tenant + weights in key)│
        │ save: per-layer gather → pinned slab (async, backpressure)
        │ load: pipelined H2D → scatter; pin by idle hint          │
        │ policy: TTL / retention / residency tier / audit log     │
        └───────┬──────────────┬──────────────┬───────────────────┘
                ▼              ▼              ▼
          pinned CPU RAM    local NVMe    object store (S3) ◄── survives pod death
                                               ▲
        fleet control plane (commercial): placement, quotas, audit UI
```

## 6. Evidence

### Proven

- Cross-instance resume: KV saved by one vLLM 0.24.0 process loaded by a separate process, prefill skipped, `identical_output=True` (OPT-125m, A10G, two Modal containers sharing a Volume, 2 external blocks); 8,176/8,192 tokens matched at 8k on Qwen3-0.6B, next-token correct. TP=1, single attention backend, single-step prefill only. Source: dexa ledger `cross-instance-resume`.
- Connector conformance: 10/10 KVConnectorBase_V1 hook signatures match vLLM 0.24.0 (signature match only; a 3-arg constructor change was found later by live probe). Source: `benchmarks/results/2026-07-09-vllm-connector-check.json`.
- Offload-restore beats cold prefill in-engine (LMCache's mechanism, not ours): CPU restore 10.0x at 4k (301 to 30 ms), 17.1x at 16k (1,320 to 77 ms), 0.875 GB at a self-reported 22.0 GB/s (Qwen2.5-7B, vLLM 0.24.0, prefix cache off, single request, LMCache version unpinned, 4k and 16k only). Source: dexa ledger `lmcache-offload-restore`.
- State must be keyed to exact weights: base<->instruct KV injection diverges within 2-5 tokens (HF-level, one passage). Source: dexa ledger `kv-interchange-weights`.
- fp8/int8 persisted KV halves bytes with 100% greedy agreement over 48 tokens; int4 diverges at token 14 (HF-level, one passage). Source: dexa ledger `kv-interchange-formats`.

### Bounded / contradicted

- Lossless persisted 8B state: bf16 blob save/resume gives identical greedy output at 8k-64k (Llama-3.1-8B, A100) but through HFBackend, not vLLM; its 3.7-5.5x is versus HF eager prefill and is superseded as a vLLM-relative speedup. Source: `2026-07-09-a100-8b-persist-native-load.json`; 04-conflicts #2.
- Restore-vs-re-prefill points both ways (CONTRADICTED): 10-34x (HF copies, vLLM prefix hit, LMCache CPU; Qwen2.5-7B; single request) versus 0.37x single-request and 1.6-2.2x worse P99 under concurrency (Llama-3.1-8B; Dexa connector from Modal Volume at ~0.6 GB/s; byte accounting inconsistent). No run measures LMCache under load in dexa; the two repos never ran the same LMCache version. Source: dexa ledger `restore-vs-reprefill-cross-repo-contradiction`; 04-conflicts #1, #4, #12.
- Whole-prompt keying FALSIFIED on `vllm bench serve prefix_repetition`: Dexa mean TTFT 3,355 ms at 9.8 req/s vs 610 ms no-cache vs 183 ms vLLM prefix cache (OPT-125m, A10G). Source: dexa ledger `vllm-bench-serve-prefix-repetition`.
- Adaptive load policy never worse than vanilla (P99 5,603 vs 5,596 ms; 9,371 vs 9,379 ms) but ships OFF and gives no gain; the crossover is calibrated to 64k (measured), 32k (default) and 8k (bench). Source: dexa ledger `connector-under-concurrency-sync-async`; 04-conflicts #12.
- Purpose-built session-granular connector held N=300 voice sessions stable (p95 316.8 ms whole-run, 336.8 ms full-load window; fails the 300 ms gate) on an L40S where LMCache 0.3.2 collapsed (25,997.7 ms default, 6,025.1 ms layerwise); idle-hint pinning passed at N=300 (272.2 ms) vs 200 for baseline, 1.5x. Single seed, synthetic traces, ~2.5k-token sessions, vLLM 0.9.2, 27.8 ms margin whole-run and 8.1 ms in the full-load window, 3 of 7 buckets above 300 ms, N=400 fails (34,048 ms), restore latency never timed. Source: voice ledger `vi-armB-voice-connector-reactive`, `vi-armD-predictive-pinning`, `vi-armD-n400-failure`.
- In-engine evidence stops at 16k: vLLM V1 crashed at >=32k in three configs (Qwen2.5-7B+YaRN); 32k-64k points are HF-only. Source: dexa ledger `vllm-warmstart-prefix-cache`; 04-conflicts #11.
- Incremental recompute: 4.6x fewer tokens reprocessed over a 20-step loop on tiny-random Llama, CPU, README-only. Source: dexa ledger `incremental-recompute`.
- Residency cost model "2-6x cheaper" is modeled with assumed rates (GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr); tiering is advisory in every deployed path. Source: dexa ledger `residency-cost-model`.

### Unproven

| Claim | Experiment that proves or kills it | Est. cost (estimates) |
|---|---|---|
| Greedy identity holds at 32k-128k for restored KV (and for vanilla prefix hits) across kernel paths | Cold vs prefix-hit vs CPU-restore vs S3-restore, Llama-3.1-8B, vLLM 0.28, token-divergence rate + KV byte diff | ~15 GPU-hours, 3 days |
| Zero recompute including decode tokens on multi-turn traces | Decode-block save on/off, count re-prefilled tokens per turn on Mooncake trace | ~15 GPU-hours, 3 days |
| Resume from pinned CPU/NVMe/S3 at parity or better at 32k-128k in-engine, under concurrency 8-32 | Crossover-surface experiment (04-conflicts #1), `scripts/modal_bench_resume.py` + contention, A100 and H100 | ~40 GPU-hours, 5 days |
| Survives a real pod kill with zero recompute on Kubernetes | Preemption Survival Score, 2 nodes, llm-d routing, SIGKILL mid-session, all baselines with remote tiers | ~30 GPU-hours, 5 days |
| MLA / hybrid-attention model support (the customers' models) | Bitwise round-trip on one MLA model and one hybrid model at TP=1 | ~20 GPU-hours, 5 days |
| TP>1 shard-consistent save/load, identical across TP=1/2/4 | Extend connector per `docs/CONNECTOR_COMPLETION.md`; two-process test | ~60 multi-GPU-hours, 10 days |
| Tenant isolation and audit add no measurable overhead | Prefix-repetition with 2 tenants; zero cross-tenant hits; TTFT delta | ~10 GPU-hours, 2 days |
| SGLang HiCacheStorage backend gives the same durability | Implement get/exist/set; repeat survival test | ~30 GPU-hours, 7 days |
| Self-hosters lose KV to restarts/preemption at a rate that matters, and buy audit of KV | 5 interviews with eviction/restart logs; spot-interruption data (none retrievable in research) | 0 GPU-hours, 6 weeks |

## 7. MVP and 6-week build plan

MVP: a pip-installable connector plus state store for vLLM 0.28 that survives a pod kill on Kubernetes with same-output resume at TP=1 on a dense GQA model, keyed by tenant and weights, with an audit log and a measured recompute-avoided metric, benchmarked on the independent harness against three named baselines. Assumes two senior engineers full-time; TP>1 and MLA support are explicitly outside the six weeks (the draft's own estimate for TP>1 was 10 days and 60 multi-GPU-hours).

- Week 1: port to vLLM 0.28 (both connectors are pinned: 0.24.0 and 0.9.2; expect drift such as the 3-arg constructor and CachedRequestData shape changes); replace whole-prompt keying with the blake2b block-chain index from `vkv/backends/voice_kv_connector.py`; extend `build_connector_meta` to `scheduled_cached_reqs` (the line-608 TODO) so chunked prefills are saved; add decode-block save. Reuse: `src/dexa/engine/vllm_connector.py`, `src/dexa/session/blob.py`, `tests/test_vllm_connector.py` (17 tests, 16 passing, 1 vllm-gated).
- Week 2: layer-pipelined async restore on side streams with the 1 GB staging backpressure from the voice connector (the voice loads are synchronous; the dexa async loads lost under load from disk); object-store tier (S3-compatible) behind the slab; header fields for weights hash, TP layout, attention backend, dtype, tenant. Reuse: `src/dexa/session/store.py`; `evals/modal_kv_interchange.py` for the fp8 option.
- Week 3: correctness week: KV-byte diff plus greedy-divergence rate at 8k/32k/128k against the vanilla control, adapting `/home/user/voice-inference/evals/modal_kv_correctness.py` (its pass criterion is loads fired, not text match; output never archived) and `scripts/modal_connector_xinstance.py`. Gate 1 evaluated here.
- Week 4: independent benchmark week: `vllm bench serve` multi-turn and Mooncake trace replay versus vanilla, OffloadingConnector (S3 tier), LMCache 0.5.x layerwise; contention rerun via `scripts/modal_bench_contention.py`; first Kubernetes kill test. Gates 2-4 evaluated.
- Week 5: governance: tenant namespace in the key, TTL/retention, append-only audit log, Prometheus metrics (hit rate, bytes by tier, measured restored tokens per request). New code; none exists.
- Week 6: Helm chart and llm-d well-lit-path integration; design-partner installs (zero exist today); control-plane skeleton reusing `dexa_platform/control` (hashed API keys, daily rollups). Placement and migration not started.

Not reused: compaction/cartridges (quality unproven), the CUA gateway, `edge/` (never deployed), `native/kvcodec` (does not compile). Deferred: `src/dexa/segment` until it runs inside vLLM.

## 8. Pricing model

Core: Apache-2.0, free, complete for a self-operating team (README rule: never cripple the core). Commercial control plane: per GPU under management per month, because the software does not serve tokens; a per-GPU count is expressible from the connector's engine registration. A savings-share alternative exists in the market (Tensormesh post-v1: GPU-hours plus 30% of estimated savings), and "recompute-avoided GPU-seconds" would make it expressible, but today the repos infer "warm" as not-first-touch and model savings from a static profile table, so savings-share is not honestly billable until the Week 5 metric reports measured restored tokens. A per-token provider can express the same layer as $0 cached input (Tensormesh) or 0.1x reads (Anthropic, OpenAI), but that is the provider's decision. Whether any self-hoster pays per-GPU-month for this is unmeasured.

## 9. Competitive facts

| Who | Ships (adjacent) | Does not ship, per the research files | Source |
|---|---|---|---|
| LMCache / Tensormesh | Apache-2.0 KV layer for vLLM/SGLang/Dynamo; standalone daemon so KV survives engine crashes; CPU/disk/Redis/Mooncake/S3/NIXL/GDS; 21 L2 adapters; P2P CPU sharing; adopters named Cohere, CoreWeave; Tensormesh $24.5M raised, SaaS with $0 cached input, Operator "coming soon" | No documented TTL/idle-window eviction (LRU/IsolatedLRU/noop); tenant isolation of cached KV not documented; Tensormesh names no customers | github.com/LMCache/LMCache; docs.lmcache.ai/mp/architecture.html; tensormesh.ai/faq |
| Mooncake Store | Transfer Engine (87-190 GB/s), approximate-LRU, 10 s leases, soft/hard pins, etcd HA, SSD offload with recovery after master restart, tenant-aware metadata (v0.3.13), SGLang backend tenant_id | Default single master is a SPOF; DFS tier "not production-ready"; soft pins not persisted across recovery | github.com/kvcache-ai/Mooncake docs |
| NVIDIA Dynamo KVBM + NIXL | G1-G4 tiers, presence/LFU offload filters, standalone kvbm pip, cache-salt-aware KV routing for tenant isolation (v1.4.0), NIXL 1.4.1 plugins, CMX flash tier (2H 2026) | No per-tier eviction algorithm detailed; no published KVBM performance numbers; disk-only mode experimental | docs.nvidia.com/dynamo KVBM design; github releases v1.4.0 |
| llm-d | Precise KV-event routing (default), tiered offload via OffloadingConnector, P2P pulls over NIXL, GKE Inference Gateway | P2P off by default; requires identical block size and hash seed | llm-d.ai/blog p2p-kv-cache-sharing; v0.9 notes |
| vLLM OffloadingConnector | In-tree, LRU/ARC, CPU primary plus FS/S3/P2P secondary; TieringManager RFC #38260 open | offload_prompt_only=true skips decode blocks; CPU staging for external storage future work; connector API "experimental" | docs.vllm.ai kv_offloading_usage; vLLM RFC #38260 |
| SGLang HiCache | L1/L2/L3 with Mooncake/3FS/NIXL/AIBrix/file; session/agent-aware hint RFCs #24656 (inactive), #27574 (open, TTL-bounded pins) | L1/L2 private per instance; L2 eviction policy undocumented | docs.sglang.io hicache_best_practices |
| FlexKV, AIBrix, Alibaba Tair KVCache | Tiered offload merged into vLLM 0.17.2+/SGLang; S3FIFO L1; Tair quota-based eviction, sold as a cloud product | Production users unknown (FlexKV); Tair pricing/engine-independence unreadable | github.com/taco-project/FlexKV; github.com/alibaba/tair-kvcache |
| Storage vendors (WEKA, VAST, DDN, MinIO, Pliops, Graid, IBM) | NVMe/object KV extenders on NIXL/Dynamo/LMCache; IBM Storage Scale 56x TTFT at 130K | No pricing found; no engine-independent multi-tenant KV service found | blocksandfiles.com GTC roundup; IBM Redbook MD260021 |
| Hosted providers | Anthropic 5 min / 1 h TTL, org-isolated caches; OpenAI 30 min / 24h; Google hourly storage billing; Fireworks x-session-affinity and x-prompt-cache-isolation-key headers | No guaranteed session-pinned KV retention SKU found | platform.claude.com; developers.openai.com; docs.fireworks.ai |

None of the projects above document versioned or branchable KV state or an audit log of state operations in the research files; whether that matters is a founder decision.

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| Greedy identity is unattainable at long context for any restore path (kernel-path numerics) | Token-divergence rate, restore vs cold, with vanilla prefix-hit control, 8k-128k | restore diverges where the vanilla control does not, on >1% of sessions | KV bytes identical on 100%, divergence <= control |
| Survival is not differentiated: baselines with remote tiers also survive the kill | Kill test on OffloadingConnector+S3, LMCache+L2, Mooncake SSD | all three match on tokens re-prefilled, correctness and P99 | product wins on >= 2 of the three |
| Zero recompute fails without decode-KV save | Tokens re-prefilled per resume, decode blocks on | > 1 block (16 tokens) per resume | <= 1 block |
| Resume from a durable tier still loses at long context | Resume/cold TTFT, pinned CPU and NVMe and S3, 8B, single request, in-engine | < 1.0x at 128k | >= 2.0x at 64k (LMCache 17.1x at 16k; llm-d 8.5x at 48k) |
| Under load the connector hurts P99 | 16 x 8192 and 8 x 32k concurrent replay per `modal_bench_contention.py` | P99 > 1.2x vanilla | P99 <= 1.0x vanilla with loads on |
| Prefix-sharing traffic gets no hits | `prefix_repetition`, 1024-token prefix, 60 prompts | mean TTFT > 1.5x vLLM prefix cache (183 ms) | within 1.2x and hit count >= vLLM's |
| Customers' models (MLA/hybrid, TP>1) need per-architecture layouts | Bitwise round-trip on one MLA and one hybrid model; TP=1/2/4 identity | not identical after 10 days each | identical on all |
| vLLM connector API drift | Breaking hook changes per quarter (0.24.0 in use 2026-07-09; 0.28.0 released 2026-08-26) | > 2 breaking changes/quarter forcing a fork | <= 1, absorbed in the shim |
| Nobody runs it | Design partners in staging after 6 weeks | 0 of 5 | >= 2 of 5 with audit log enabled |
| The pain is hosted-TTL pain, not self-hoster pain | 5 interviews with restart/eviction logs; KV lifetime vs idle-gap distribution on a real fleet | 0 of 5 report KV loss to restarts/preemption | >= 2 of 5 with logs |

## 11. Founder decisions

- Whether LMCache/Tensormesh, Mooncake, KVBM and the in-tree OffloadingConnector sharing the seam disqualifies the direction or defines the baselines. Evidence: the Week 4 parity benchmark.
- Market size of governance-first buyers (regulated, sovereign, neocloud). Unmeasured; five-partner outreach is the experiment.
- Licensing and monetization: Apache-2.0 open-core with a per-GPU-month control plane (the draft's choice) vs source-available/BSL vs savings-share vs hosted-only. The README declares hosted SaaS out of scope; `edge/` and `dexa_platform` build exactly that (04-conflicts #7).
- Positioning: durability/governance-first (the draft's choice) vs a speed/cost positioning that the ledger currently contradicts.
- Model class and TP: dense GQA 7-8B at TP=1 first (the draft's scope, matching the evidence) vs the MoE/MLA/hybrid models at TP=8 that the workload data shows customers run. Evidence: the MLA/hybrid round-trip experiment.
- Own the durable store vs build on LMCache's MP server / Mooncake Store / NIXL as the tier and sell only policy, keying and control plane. Evidence: transfer-rate measurements (22 GB/s pinned CPU vs ~0.6 GB/s Volume) and the API-drift count.
- Out-of-tree connector vs fork vLLM for scheduler-native prefetch and eviction ordering (the API cannot express request-less prefetch) vs contribute to vLLM's TieringManager RFC and SGLang's session RFCs.
- Engine order: vLLM-first (measured) vs SGLang (HiCacheStorage, unmeasured) vs Dynamo.
- Distribution: Kubernetes/llm-d well-lit path (the draft's choice) vs pip-only vs Dynamo/NIXL plugin.
- GPU class: all evidence is A100/L40S/A10G; the L40S wall (N=300 needs ~42 GB vs ~26 GB pool) may not exist on 80 GB parts, and FP8 KV halves bytes (unmeasured; both repos name this their top threat).
- Proof metric: Preemption Survival Score (the draft's choice) vs a measured $/GPU-hour-saved metric a buyer pays on.
- Go/no-go thresholds: "2 of 5 partners with audit log enabled" and the kill-gate numbers above are the draft's; the founder sets them.
- Whether the mutation/versioning DAG (HF-only today, the README's stated wedge) is the product or a later substrate.

## 12. Combinations

- With the voice session-aware residency candidate: arm D's idle-hint pinning (`kv_transfer_params.idle_ms`, ~40 lines in the connector) is a residency policy that plugs into this layer's policy slot; its harness becomes this product's load generator.
- With a hosted session-stateful agent provider: this layer is the substrate; the provider is the BYOC-to-hosted upsell the README excludes and `edge/` prototypes.
- With llm-d / Kubernetes gateway packaging: the audit log and tenant keying are a compliance layer llm-d's routing does not document; a well-lit path is the distribution channel.
- With compaction/cartridges: only as a later "shrink the persisted object" lever; AM loses to H2O at 128x multi-needle (0.53 vs 0.82) and cartridge quality is unproven.
