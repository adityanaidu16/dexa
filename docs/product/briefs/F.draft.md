## 1. Thesis

Candidate F is an Apache-2.0 infrastructure layer that turns an inference engine's KV cache into a durable, portable, governed object owned by the operator: session state that survives a pod restart, a replica move or a spot preemption and resumes on a different vLLM process with bit-identical output and no re-prefill beyond block alignment; keyed to exact weights and tenant; audited, retained and evicted by policy rather than by whatever LRU the engine happens to run. It is sold to platform teams, neoclouds and regulated deployments that run vLLM or SGLang themselves, with a commercial fleet control plane (placement, tenant policy, audit, quotas) as the paid layer. The sentence a customer would repeat: "When our agent pod dies, the session comes back on another node with the same output and zero recompute, and we can prove which tenant's context was touched." The measured foundation is narrow and real: cross-instance resume is PROVEN at TP=1 on a real vLLM 0.24.0, and the two repos hold two working KVConnector implementations. The speed story is not yet the product; durability, portability and governance are.

## 2. Customer and workload

Who buys: the ML-platform team that already operates vLLM (PyPI 0.28.0, Aug 26 2026) or SGLang (0.5.18) on Kubernetes, often via llm-d (CNCF Sandbox, v0.9.0 Aug 17 2026); neoclouds selling dedicated capacity (Tensormesh resells reserved H200s at $2.50/hr; Wafer cites MI355X rental at $2.29-2.95/GPU-hr); and regulated deployments that must keep inference state inside their perimeter.

What they run: long-horizon coding and computer-use agents on open-weight models. TraceLab's ~4,300 Claude Code/Codex sessions: median cached prefix 115,584-126,180 tokens per step, ~857-886 new input tokens, 184-252 output tokens; human gap median 1.4 min, p90 20.6 min; prefix-cache hits 95.7% overall but 84.4% on user-initiated steps. llm-d's 219 production Claude Code sessions on GLM-5.2: median 195K input / 317 output tokens, 96% of requests reuse >=90% of input verbatim, inter-turn pauses median 2 s and p99 11.4 min, and "TP=8 keeps eight copies of the KV cache". Codex traces replayed by vLLM/Mooncake: ~131:1 input:output, inter-turn delays 5.2 s median to 81.4 s P99. Alibaba's production traces: ideal hit ratios 54-62%, P99 KV lifespan 97 s in to-B traffic. The repos' own measurements cover only Llama-3.1-8B and Qwen2.5-7B on A100-80GB, A10G and L40S at 4k-64k, TP=1.

How they pay: GPU-hours, not tokens; re-prefill never appears on an invoice, it appears as prefill throughput (arm A collapse on an L40S: 9,466-10,779 prompt tok/s while the prefix-cache hit rate fell from ~92% to 1.6-1.9%). Hosted comparators: Anthropic's 5-minute default TTL (1 hour at 2x write price), OpenAI's 30-minute (GPT-5.6+) or up-to-24h retention, DeepSeek's disk cache cleared "within a few hours to a few days".

## 3. The pain, in the customer's words

No customer interviews exist in either repo; these are statements from public issues and docs:

- "After a gap longer than the TTL, the next request recomputes the full input and re-establishes the cache" (Claude Code docs). A Codex user on Azure measured 3.28M re-billed tokens, 71% of fresh input, once a session passed ~150K tokens.
- "Host memory cannot be pooled across instances or across hosts, not even for two instances on the same node" (SGLang HiCache best practices).
- "Peers must share identical block size and hash seed or transfers silently drop to zero" (llm-d P2P KV sharing).
- "This API is experimental and subject to change" (vLLM KVConnectorBase_V1 docstring, unchanged since the April 2025 merge).
- Building the equivalent in-house "takes 20 engineers and three or four months" (Tensormesh CEO).
- From the repos: "Memory your engine doesn't know about will kill it" (voice-inference FINDINGS), after connector staging buffers outside vLLM's budget OOM'd the engine at 345 concurrent requests.

## 4. Value proposition and the proof-of-value benchmark

The Morph-style proof for this candidate is a correctness-and-durability benchmark, not a TTFT race, because the ledger already shows where a TTFT race goes: the Dexa connector loading raw KV from a Modal Volume lost to vLLM re-prefill at Llama-3.1-8B/8k (1,681 ms vs 617 ms, 0.37x), and under 16-24-way concurrency its P99 TTFT was 1.6-2.2x worse than vanilla even with async loading. 04-conflicts reconciles every restore number in both repos with one rule: restore wins only when the tier's available bandwidth times the engine's prefill time exceeds the KV bytes; pinned RAM at ~10-20 GB/s crosses over below 4k single-request, the connector's disk path at ~0.6 GB/s only near 64k. The honest claim is "zero recompute, identical output, on any replica", with speed parity as a gate, not a headline.

Metric: the Preemption Survival Score.

Setup: two vLLM instances (same weights hash) behind round-robin routing that deliberately defeats local prefix caching (BENCHMARK_PLAN gotcha #1). Drive `vllm bench serve` multi-turn with `--max-active-conversations` forcing eviction and retrieval, plus the Mooncake `conversation_trace` replayed via `--dataset-name timed_trace --self-timed`. Mid-run, SIGKILL instance A after turn k and reschedule its sessions on B.

Three numbers per configuration: (1) fraction of resumed sessions whose greedy continuation is byte-identical to an uninterrupted run; (2) tokens re-prefilled per resume (block-alignment floor; the ledger measured 8,176 of 8,192 tokens matched, i.e. 16 re-prefilled); (3) P50/P99 TTFT after resume versus cold re-prefill of the full history at 8k, 32k, 64k and 128k.

Baselines, named: vanilla vLLM prefix caching (state dies with the pod); vLLM's native OffloadingConnector with CPU primary and FS/S3 secondary tier (LRU/ARC; offload_prompt_only defaults true, so decode blocks are skipped); LMCache 0.5.4 with a remote L2 (S3/Redis/Mooncake) and with local CPU only; Mooncake Store with SSD offload; llm-d P2P (CPU-to-CPU over NIXL, off by default); SGLang HiCache with a Mooncake L3 once an SGLang backend exists.

Targets, and why a skeptic would believe them: identical output on 100% of resumed sessions (held at TP=1 on OPT-125m and Qwen3-0.6B; TP>1 is the long pole and unproven). Resume TTFT at 128k at least at parity with re-prefill (kill gate below). External anchors put the pinned-CPU crossover between 8k and 48k: llm-d measured a 48K-token pull at 235 ms vs 1,988 ms recompute on gpt-oss-120b and cites a ~8K-12K-token crossover for GLM-5.2-FP8; the LMCache paper says remote storage beats prefill only above ~256K tokens at 32 Gbps. The ledger's 0.37x came from ~0.6 GB/s Volume loads versus LMCache's 22.0 GB/s pinned-CPU retrieval, so the first job is to change the tier, not the thesis. The skeptic believes it because the harness is not ours, the correctness test is bitwise, and the repo already published its losing number.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| vLLM (0.24-0.28) / SGLang | OSS, unmodified | Serves tokens; owns HBM; exposes KVConnectorBase_V1 / HiCacheStorage |
| State connector (`src/dexa/engine/vllm_connector.py`, 917 lines; `vkv/backends/voice_kv_connector.py`, 574 lines) | Custom | Scheduler+worker hooks: block-hash keyed lookup, chunked-prefill-aware save, async load on side streams, pin-by-hint via `request_finished` |
| State object format (`src/dexa/session/blob.py`, DEXAKV01) | Custom | bf16-native, mmap-loadable, block-granular; header carries weights hash, TP layout, attention backend, dtype, tenant |
| Pinned-CPU slab + NVMe + object tiers | Custom policy over OSS storage (S3, NVMe; optionally NIXL/Mooncake transport) | Durable copy survives pod death; placement by policy, not engine LRU |
| Governance: tenant namespace, TTL/retention, audit log, recompute-avoided metrics | Custom; nothing exists today (grep for rbac/audit/prometheus returns only comments) | The v1 differentiator for regulated buyers |
| Segment DAG, incremental recompute, RoPE re-phase (`src/dexa/segment`, 643 lines) | Custom, HF-backend only | Later substrate for edit-in-place sessions; not in the vLLM path |
| Benchmark harness (`vkv/*` ~2,000 lines; `benchmarks/vllm_connector_check.py`; `scripts/modal_bench_contention.py`) | Custom | Independent-harness replays, conformance tiers, contention runs |
| Fleet control plane (commercial) | Custom; `dexa_platform/control` (620 lines) is a keys/metering scaffold | Session-to-replica placement, KV migration, quotas, dashboards |

Where the differentiation lives: not in kernels, the engine or the transport (all OSS). It lives in the connector's keying and save semantics, the state-object format (portable across processes and, once built, across TP layouts), the durability policy (what is guaranteed to survive) and the control plane (who may read which state, for how long, with what audit). vLLM and SGLang are used, not replaced; the connector seam is the one LMCache, NIXL, Mooncake and FlexKV also use.

```
 agent harness ── OpenAI-compatible API ──► vLLM / SGLang (unchanged)
                                               │ KVConnector / HiCacheStorage
                                               ▼
        ┌───────────── Dexa state layer (Apache-2.0) ─────────────┐
        │ block-hash index (blake2b chain; tenant + weights in key)│
        │ save: per-layer gather → pinned slab (async, backpressure)
        │ load: contiguous H2D → scatter; pin by idle hint         │
        │ policy: TTL / retention / residency tier / audit log     │
        └───────┬──────────────┬──────────────┬───────────────────┘
                ▼              ▼              ▼
          pinned CPU RAM    local NVMe    object store (S3) ◄── survives pod death
                                               ▲
        fleet control plane (commercial): placement, quotas, audit UI
```

## 6. Evidence

### Proven

- Cross-instance resume: KV saved by one vLLM 0.24.0 process loaded by a separate process, prefill skipped, `identical_output=True` (OPT-125m, A10G, two Modal containers); 8,176/8,192 tokens matched at 8k on Qwen3-0.6B. Source: dexa ledger `cross-instance-resume`; `docs/RESULTS.md` "Cross-instance confirmed".
- Connector conformance: 10/10 KVConnectorBase_V1 hook signatures match vLLM 0.24.0. Source: `benchmarks/results/2026-07-09-vllm-connector-check.json`.
- Lossless persisted state on a real 8B: bf16 blob save/resume bit-identical at 8k-64k (Llama-3.1-8B, A100; its 3.7-5.5x is versus HF eager prefill and not cited as a speedup). Source: `benchmarks/results/2026-07-09-a100-8b-persist-native-load.json`; 04-conflicts #2.
- Offload-restore beats cold prefill in-engine: LMCache CPU restore 10.0x at 4k (301 to 30 ms), 17.1x at 16k (1,320 to 77 ms), 0.875 GB moved at 22.0 GB/s (Qwen2.5-7B, vLLM 0.24.0, prefix cache off, single request). Source: dexa ledger `lmcache-offload-restore`.
- fp8/int8 persisted KV halves bytes with 100% greedy agreement over 48 tokens; int4 diverges at token 14. Source: dexa ledger `kv-interchange-formats`.
- State must be keyed to exact weights: base<->instruct KV injection diverges within 2-5 tokens. Source: dexa ledger `kv-interchange-weights`.
- A purpose-built session-granular connector held N=300 voice sessions stable (p95 316.8 ms whole-run, 336.8 ms in the full-load window) on an L40S where LMCache 0.3.2 collapsed (25,997.7 ms default, 6,025.1 ms layerwise); idle-hint pinning passed the 300 ms gate at N=300 (272.2 ms) vs 200 for baseline. Source: voice ledger `vi-armB-voice-connector-reactive`, `vi-armD-predictive-pinning`.

### Bounded / contradicted

- Restore-vs-re-prefill points both ways (CONTRADICTED): 10-34x (HF copies, vLLM prefix hit, LMCache CPU; Qwen2.5-7B; single request) versus 0.37x single-request and 1.6-2.2x worse P99 under concurrency (Llama-3.1-8B; Dexa connector from Modal Volume at ~0.6 GB/s, byte accounting inconsistent). No run measures LMCache under load in dexa; the two repos never ran the same LMCache version. Source: dexa ledger `restore-vs-reprefill-cross-repo-contradiction`; 04-conflicts #1, #4, #12.
- Whole-prompt keying got zero hits on `vllm bench serve prefix_repetition`: Dexa mean TTFT 3,355 ms at 9.8 req/s vs 610 ms no-cache vs 183 ms vLLM prefix cache (OPT-125m, A10G). Source: dexa ledger `vllm-bench-serve-prefix-repetition`.
- Adaptive load policy is never worse than vanilla (P99 5,603 vs 5,596 ms; 9,371 vs 9,379 ms) but ships OFF and gives no gain; the crossover is calibrated to 64k (measured), 32k (default) and 8k (bench). Source: dexa ledger `connector-under-concurrency-sync-async`; 04-conflicts #12.
- In-engine evidence stops at 16k: vLLM V1 crashed at >=32k in three configs; the 32k-64k points are HF-only. Source: dexa ledger `vllm-warmstart-prefix-cache`; 04-conflicts #11.
- Incremental recompute: 4.6x fewer tokens reprocessed over a 20-step loop on tiny-random Llama, CPU; no GPU wall time. Source: dexa ledger `incremental-recompute`.
- Voice 1.5x: single seed, 27.8 ms margin whole-run, 8.1 ms in the full-load window; 3 of 7 buckets above 300 ms; N=400 fails (34,048 ms). Source: voice ledger `vi-armD-predictive-pinning`, `vi-armD-n400-failure`.
- Residency cost model "2-6x cheaper" is modeled with assumed rates (GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr); the tiering policy is advisory in every deployed path. Source: dexa ledger `residency-cost-model`; 04-conflicts #3.

### Unproven

| Claim | Experiment that proves or kills it | Est. cost (estimates, not measured) |
|---|---|---|
| Resume from pinned CPU/NVMe beats re-prefill at 32k-128k with block-hash keying, in-engine | `scripts/modal_bench_resume.py` with pinned-slab tier and chunked prefill at 8k/32k/64k/128k, 8B, on A100 and H100 (04-conflicts #1 crossover-surface experiment) | ~40 GPU-hours, 5 days |
| TP>1 shard-consistent save/load with bit-identical output across TP layouts | Extend connector per `docs/CONNECTOR_COMPLETION.md` "TP>1"; two-process test at TP=2 and TP=4 | ~60 multi-GPU-hours, 10 days |
| Survives pod kill with zero recompute on Kubernetes | Preemption Survival Score on 2 nodes, llm-d routing, SIGKILL mid-session | ~30 GPU-hours, 5 days |
| Parity with LMCache / OffloadingConnector / vanilla on Mooncake traces | `docs/BENCHMARK_PLAN.md` first experiment, connector swapped; never run | ~50 GPU-hours, 7 days |
| Tenant isolation and audit add no measurable overhead | Prefix-repetition run with 2 tenants, identical prompts; assert zero cross-tenant hits; TTFT delta | ~10 GPU-hours, 2 days |
| Buyers will run it in staging | 5 design-partner deployments (zero exist) | 0 GPU-hours, 6 weeks |
| SGLang HiCacheStorage backend gives the same durability | Implement get/exist/set; repeat survival test | ~30 GPU-hours, 7 days |

## 7. MVP and 6-week build plan

MVP: a pip-installable connector plus state store for vLLM that survives a pod kill on Kubernetes with bit-identical resume, keyed by tenant and weights, with an audit log and a recompute-avoided metric, benchmarked on the independent harness against three named baselines.

- Week 1: replace whole-prompt keying with the blake2b block-chain index from `vkv/backends/voice_kv_connector.py`; extend `build_connector_meta` to `scheduled_cached_reqs` (the line-608 TODO) so chunked prefills are saved; port the pinned-slab async save with 1 GB staging backpressure. Reuse: `src/dexa/engine/vllm_connector.py`, `src/dexa/session/blob.py`, `tests/test_vllm_connector.py` (16 passing).
- Week 2: object-store tier (S3-compatible) behind the slab; header fields for weights hash, TP layout, attention backend, dtype, tenant; fp8 storage option. Reuse: `src/dexa/session/store.py`; `evals/modal_kv_interchange.py` for the fp8 check.
- Week 3: TP>1 shard-consistent layout (long pole per `docs/CONNECTOR_COMPLETION.md`); bitwise correctness suite from `evals/modal_kv_correctness.py` plus the two-process test in `scripts/modal_connector_xinstance.py`.
- Week 4: independent benchmark week: `vllm bench serve` multi-turn and Mooncake trace replay versus vanilla, OffloadingConnector, LMCache; contention rerun via `scripts/modal_bench_contention.py`. Section 10 gates are evaluated here.
- Week 5: governance: tenant namespace in the key, TTL/retention policy, append-only audit log, Prometheus metrics (hit rate, bytes by tier, recompute-avoided GPU-seconds). New code; none exists today.
- Week 6: Helm chart and llm-d well-lit-path integration; design-partner install; control-plane skeleton reusing `dexa_platform/control` (hashed API keys, daily rollups). Placement and migration are not started.

Not reused: compaction/cartridges (quality unproven), the CUA gateway, `edge/` (never deployed), `native/kvcodec` (does not compile). Deferred: the `src/dexa/segment` mutation DAG until it runs inside vLLM.

## 8. Pricing model

Core: Apache-2.0, free, complete for a self-operating team (README rule: never cripple the core). Commercial control plane: billed per GPU under management per month, because the software does not serve tokens; a per-GPU fee is expressible from the connector's engine registration. A savings-share alternative exists in the market (Tensormesh's post-v1 formula: GPU-hours plus 30% of estimated savings), and the README's headline metric "recompute-avoided GPU-seconds" would make it expressible, but today the repos infer "warm" as not-first-touch and model savings from a static profile table, so savings-share is not yet honestly measurable until the Week 5 metric reports measured restored tokens per request. A per-token provider can express the same layer as $0 cached-input pricing (Tensormesh bills cached input at $0; Anthropic and OpenAI read at 0.1x), but that is the provider's decision.

## 9. Competitive facts

| Who | Ships (adjacent) | Does not ship, as far as the research files show | Source |
|---|---|---|---|
| LMCache / Tensormesh | Apache-2.0 KV layer for vLLM/SGLang/Dynamo; CPU/disk/Redis/Mooncake/S3/NIXL/GDS backends; 21 L2 adapters; P2P CPU sharing; $24.5M raised; SaaS with $0 cached input; Operator "coming soon" | No documented TTL/idle-window eviction (LRU/IsolatedLRU/noop only); tenant isolation of cached KV not documented; no named customers | github.com/LMCache/LMCache; docs.lmcache.ai/mp/architecture.html; tensormesh.ai/faq |
| Mooncake Store | Transfer Engine (87-190 GB/s), approximate-LRU store, 10 s leases, soft/hard pins, etcd HA, SSD offload, tenant-aware metadata (v0.3.13) | Default single master is a SPOF; DFS tier "not production-ready"; OffsetAllocator truncates its file on restart; soft pins not persisted across recovery | github.com/kvcache-ai/Mooncake docs |
| NVIDIA Dynamo KVBM + NIXL | G1-G4 tiers, presence/LFU offload filters, standalone kvbm pip, NIXL 1.4.1 plugins, CMX/BlueField-4 flash tier (2H 2026) | No per-tier eviction algorithm detailed; no published KVBM performance numbers; disk-only mode experimental | docs.nvidia.com/dynamo KVBM design; pypi kvbm |
| llm-d | Precise KV-event routing (default), tiered offload via vLLM OffloadingConnector, P2P pulls over NIXL, GKE Inference Gateway | P2P off by default; requires identical block size and hash seed | llm-d.ai/blog p2p-kv-cache-sharing; v0.9 notes |
| vLLM OffloadingConnector | In-tree, LRU/ARC, CPU primary plus FS/S3/P2P secondary; TieringManager RFC #38260 open | offload_prompt_only=true skips decode blocks; CPU staging for external storage is future work; connector API "experimental" | docs.vllm.ai kv_offloading_usage; vLLM RFC #38260 |
| SGLang HiCache | L1/L2/L3 with Mooncake/3FS/NIXL/AIBrix/file; session/agent-aware hint RFCs #24656, #27574 | L1/L2 private per instance; L2 eviction policy undocumented; agent-aware RFC labeled inactive | docs.sglang.io hicache_best_practices |
| FlexKV, AIBrix, Alibaba Tair KVCache | Tiered offload merged into vLLM 0.17.2+/SGLang; S3FIFO L1; Tair quota-based eviction, sold as a cloud product | Production users unknown (FlexKV); Tair pricing/engine-independence unreadable | github.com/taco-project/FlexKV; github.com/alibaba/tair-kvcache |
| Storage vendors (WEKA, VAST, DDN, MinIO, Pliops, Graid, IBM) | NVMe/object KV extenders on NIXL/Dynamo/LMCache; e.g. IBM Storage Scale 56x TTFT at 130K | No pricing found; no engine-independent multi-tenant KV service found anywhere | blocksandfiles.com GTC roundup; IBM Redbook MD260021 |
| Frontier providers | Anthropic 5 min / 1 h TTL, org-isolated caches; OpenAI 30 min / 24h; Google hourly storage billing | No guaranteed session-pinned KV retention SKU found | platform.claude.com prompt-caching; developers.openai.com prompt-caching |

None of the projects above document versioned or branchable state or an audit log of state operations in the research files; whether that matters is a founder decision.

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| Resume from a durable tier still loses to re-prefill at long context | Resume TTFT / cold re-prefill TTFT, pinned CPU and NVMe, 8B and TP=2, single request, in-engine | < 1.0x at 128k | >= 2.0x at 64k (LMCache reached 17.1x at 16k; llm-d 8.5x at 48k) |
| Under load the connector hurts P99 | 16 x 8192-token concurrent replay per `modal_bench_contention.py` | P99 > 1.2x vanilla | P99 <= 1.0x vanilla with loads on |
| Prefix-sharing traffic gets no hits | `prefix_repetition`, 1024-token prefix, 60 prompts | mean TTFT > 1.5x vLLM prefix cache (183 ms) | within 1.2x and hit count >= vLLM's |
| TP>1 cannot be made shard-consistent without forking | Bitwise resume across TP=1/2/4 layouts | not bit-identical after 10 days | identical on all three |
| vLLM connector API drift | Breaking hook changes per quarter (0.24.0 in use 2026-07-09; 0.28.0 released 2026-08-26) | > 2 breaking changes/quarter forcing a fork | <= 1, absorbed in the shim |
| Nobody runs it | Design partners in staging after 6 weeks | 0 of 5 | >= 2 of 5 with audit log enabled |
| Preemption survival is not zero-recompute | Tokens re-prefilled per resume on the K8s kill test | > 1 block per resume | <= 1 block (16 tokens) |

## 11. Founder decisions

- Whether LMCache/Tensormesh, Mooncake, KVBM and the in-tree OffloadingConnector sharing the connector seam disqualifies the direction or defines the baselines. Evidence: the Week 4 parity benchmark, and whether any of them ship durability, tenant keying and audit by then.
- Market size of governance-first buyers (regulated, sovereign, neocloud). Unmeasured; five-partner outreach is the experiment.
- Hosted control plane vs BYOC vs pure OSS. The README declares hosted multi-tenant SaaS out of scope; `edge/` and `dexa_platform` build exactly that (04-conflicts #7). Evidence: which partners pay for what.
- Build the durable tier on LMCache's L2 adapters / NIXL / Mooncake as transport, or own the store. Evidence: transfer-rate measurements (22 GB/s pinned CPU vs ~0.6 GB/s Volume) and the API-drift count.
- Engine order: vLLM-first (measured) vs SGLang (HiCacheStorage, unmeasured) vs Dynamo.
- GPU and model class: all evidence is 7-8B on A100/L40S at TP=1; the L40S wall (N=300 needs ~42 GB vs ~26 GB pool) may not exist on 80 GB parts (both repos name this their top threat, unmeasured).
- Whether to fork vLLM for scheduler-native prefetch and eviction ordering (the KVConnector API cannot express request-less prefetch) or stay out-of-tree.
- Whether the mutation/versioning DAG (HF-only today) is the product or a later substrate.
- Pricing: per-GPU-month vs savings-share vs $0-cached-token; decides which metric is built first.
- Upstream posture: contribute to vLLM's TieringManager RFC and SGLang's session RFCs, or differentiate out-of-tree.

## 12. Combinations

- With the voice session-aware residency candidate: arm D's idle-hint pinning (`kv_transfer_params.idle_ms`, ~40 lines in the connector) is a residency policy that plugs into this layer's policy slot; its harness becomes this product's load generator.
- With a hosted session-stateful agent provider: this layer is the substrate; the provider is the BYOC-to-hosted upsell the README excludes and `edge/` prototypes.
- With llm-d / Kubernetes gateway packaging: the audit log and tenant keying are the compliance layer llm-d's routing does not document; a well-lit path is the distribution channel.
- With compaction/cartridges: only as a later "shrink the persisted object" lever; AM loses to H2O at 128x multi-needle (0.53 vs 0.82) and cartridge quality is unproven.