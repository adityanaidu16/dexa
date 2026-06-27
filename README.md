# Dexa

Open-source inference-state engine. Turns ephemeral KV cache into persistent, mutable, versioned, governed state.

> **Implementation status (v0.1).** The current prototype is **compaction-first**:
> it treats a chunk of context as a *compact* KV cache (much smaller keys/values +
> per-key attention biases) that preserves the model's behavior. This is the
> [Attention Matching](https://arxiv.org/abs/2602.16284) / Cartridges / STILL line
> of work, and it directly attacks the memory wall and long-horizon agentic
> context. Built so far: a model-backend abstraction (HF backend with an *exact*
> compact-decode path; a vLLM adapter for the cluster), the Attention Matching
> compactor + selection baselines (H2O/SnapKV/recent/random), a STILL-style
> amortized perceiver, a bounded iterative **working memory** for agent loops, an
> LMCache-style reuse+tiering baseline, and a benchmark harness. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
> [`docs/RESULTS.md`](docs/RESULTS.md) (honest results, incl. where the win is and
> isn't). Headline: on SmolLM2-360M needle-recall, Attention Matching is the only
> method that survives **128× compression** (recall 0.90 vs H2O 0.43), while at
> moderate ratios all good methods tie. The DAG / segment-dependency design below
> is the original framing and a candidate later substrate, not what's built today.

## What it is

Dexa is a library + sidecar that interposes between your inference engine (vLLM, SGLang, Dynamo, TensorRT-LLM) and the storage hierarchy. It does not serve tokens — it owns the lifecycle of the engine's state.

```
  Agent harness / app
        │  (unchanged: OpenAI-compatible API to the engine)
        ▼
  Inference engine (vLLM / SGLang / Dynamo / TRT-LLM)
        │  KV-connector interface  ◄── Dexa hooks here
        ▼
  ┌──────────────────────────────────────────────┐
  │  DEXA                                         │
  │  - extract / load KV (prefill & decode)       │
  │  - segment-level dependency tracking          │
  │  - incremental recompute on mutation          │
  │  - versioning / branching / diff / rollback   │
  │  - tiering & placement across the hierarchy   │
  │  - state-level observability                  │
  └──────────────────────────────────────────────┘
        ▼
  Storage tiers: GPU HBM → CPU DRAM → NVMe → network/object store
```

## The problem

KV cache is created during prefill, used for a single decode pass, and discarded. That breaks badly for modern workloads:

- **Recompute waste.** System prompts, tool definitions, and documents re-encoded from scratch on every session. Up to ~40% of prefill compute is redundant.
- **Mutation invalidation.** When an agent edits a mid-context segment (a file, tool result, retrieved doc), every cached token downstream is invalidated and recomputed — ~90% slowdowns and O(N²) blowups in long-running agent loops.
- **Memory-wall pressure.** A 128K-token context consumes ~40GB of KV. Long-horizon and multimodal workloads push context faster than per-token optimizations absorb.
- **Fragmentation.** Each engine, replica, and node manages KV in isolation. Identical context processed on a different worker can't be reused.

Existing OSS work (vLLM prefix caching, LMCache offloading/sharing) solves reuse of static prefixes. It does not treat state as mutable, versioned, or governed. That's the gap Dexa fills.

## Core capabilities (v1)

### Persistent KV lifecycle
- Extract KV after prefill, load before decode, persist across requests, sessions, and replicas.
- Tiered placement across GPU/CPU/NVMe/network with configurable eviction.
- Survives restarts and spot interruptions when backed by a shared tier.

### Mutable state with incremental recompute
- Segment-level dependency tracking: the cache is structured so the system knows which downstream segments depend on which upstream ones.
- On a mid-context edit, recompute only the affected segment(s) and everything causally downstream — not the entire prefix.
- The thing nothing in open-source does today.

### Versioning & branching
- `commit` a turn's delta and version the result; `branch` a session for sub-agents or speculative paths; `diff` and `rollback`.
- Multi-agent systems fork shared context without duplicating full prefills.

### Governance primitives
- Per-tenant isolation, RBAC on state operations, TTL/retention/eviction policies.
- Audit logging of every read/write/delete.
- Data residency controls; everything runs inside the operator's perimeter.

### State-level observability
- Hit rate, staleness, storage utilization by tier and model.
- Headline metric: **recompute-avoided** (GPU-seconds and tokens saved).
- Exportable to Prometheus/Grafana.

## Who this is for

- Platform/ML-infra teams running vLLM, SGLang, Dynamo, or TensorRT-LLM in production.
- Neoclouds and GPU providers serving high-volume agentic workloads.
- Regulated and sovereign deployments (finance, healthcare, public sector) that must keep inference state inside their perimeter.

What unites them: they run their own inference, they have raw-KV access, and they want infrastructure they operate — not a feature locked inside someone else's platform.

## Build sequencing

### Phase 0 — Persistent KV on one engine
- vLLM integration via KV-connector interface.
- Extract/load KV, tier across GPU → CPU → NVMe, survive restarts.
- Define the state-object format v0.
- **Gate:** match or beat LMCache on TTFT and cross-replica reuse at 128K context.

### Phase 1 — Mutable state + incremental recompute
- Segment-level dependency tracking (causal DAG over context segments).
- Incremental recompute on mid-context edits.
- Handle file/tool-result edits, replayed-message divergence, header churn.
- **Gate:** order-of-magnitude reduction in tokens reprocessed per mutating turn vs. full-reprefill baseline on an agent loop benchmark.

### Phase 1.x — Versioning, second engine, observability
- `commit` / `branch` / `diff` / `rollback` on top of the segment model.
- SGLang integration.
- Recompute-avoided dashboard.

### Phase 2 — Governance & sovereignty
- Per-tenant isolation, RBAC, TTL/retention, audit logging, encryption-at-rest.
- **Gate:** a regulated or neocloud team runs it in production inside their perimeter.

### Phase 3 — Distilled & portable state objects
- Cartridges (gradient-optimized compact KV) for high-reuse static context.
- Attention matching (analytic compression) for moderately reused context.
- Selection policy: cartridge vs. attention-matching vs. keep-raw-KV based on reuse frequency and mutation rate.

### Phase 4 (optional, commercial) — Fleet control plane
- Multi-cluster routing/placement, fleet-wide policy, managed option, SLA support.

## What's out of scope

- Closed-provider support (Anthropic/OpenAI/Gemini) — no raw KV access.
- Universal cross-model-family portability — model-relative only.
- Hosted multi-tenant SaaS control plane — operated by the customer.

## Open-source

Apache-2.0. The core stays genuinely complete for self-operating teams. A commercial fleet control plane (Phase 4) is the optional paid layer — never cripple the core.

## Competitive positioning

| Tool | Capability |
|------|-----------|
| LMCache / Tensormesh | KV offload/share/tier — append-only reuse |
| vLLM prefix caching / SGLang RadixAttention | Engine-native, single-node, static-prefix |
| llm-d (Red Hat) | KV-aware routing |
| **Dexa** | Mutable, versioned, governed state engine — engine-agnostic |

The wedge: mutation + versioning + governance as first-class, not bolted on.

## Key risks

- **KV-connector surface instability.** vLLM/SGLang interfaces change frequently. Mitigation: thin, well-tested abstraction layer.
- **Engine-native absorption.** vLLM could ship segment-level invalidation. Mitigation: anchor on versioning + governance, which are out of scope for a cache.
- **Segment dependency tracking complexity.** The hardest technical bet. Validate early with a standalone prototype.
- **Open-core monetization.** Too thin a commercial layer → no business; too thick → no adoption. Fleet/scale/compliance, never core correctness.

## License

Apache-2.0