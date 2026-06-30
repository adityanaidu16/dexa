# Dexa

**Persistent, portable inference state for open-model agents.** Keep an agent's KV cache as a server-side object, resume it on *any* GPU — after a restart, a replica move, or a spot preemption — with **~0 re-prefill and identical output**, and compact the cold tail so long sessions stay cheap. Built for open models on vLLM: the statefulness frontier providers give you for free, and self-hosters don't have.

> **Status (v0.1, honest).** The wedge — **persistent + portable session state** — is validated end-to-end: resuming a session from saved state is **14–25× faster than re-prefilling, bit-identical (lossless), and survives a full process restart** (save in one process, resume in a fresh one). Measured on SmolLM2/CPU; the gap grows with context length and model size. Try it in 2 commands below.
>
> Compaction (Attention Matching / Cartridges / STILL — [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/CARTRIDGES.md`](docs/CARTRIDGES.md)) is built and benchmarked, but it is the **optimization that shrinks the persisted state**, not the headline: we found high-ratio compaction loses multi-fact fidelity, so the design keeps recent context raw and compacts only the cold tail. Productizing the stateful path *inside* vLLM is the next build. The DAG / mutation design further down is the original vision and a later substrate.

## Why

Frontier APIs (Anthropic/OpenAI) hand agents prompt caching and memory for free. Open-model self-hosters get **ephemeral, per-instance** prefix caching — it can't survive a restart, move across replicas, or ride cheap spot capacity, and it re-prefills on every cache miss. For long-horizon agents (a coding agent accumulates 100k+ tokens of repo + tool output per session), re-paying for that context is the dominant cost. Dexa makes the session a **persistent, portable object the operator owns** — which also lets it run on the cheapest interruptible compute, because the state survives preemption.

## Quickstart

```bash
pip install -e '.[hf,bench]'          # on a CUDA box reuse the base torch; see docs/CLUSTER.md

# benchmark: resume-from-state vs cold re-prefill, across context lengths
python benchmarks/persist_demo.py bench --model <hf-model> --lengths 2000,8000,32000

# the "survives a restart" demo — two separate processes:
python benchmarks/persist_demo.py save   --model <hf-model> --length 8000 --session demo
#   ...restart the box / move GPUs...
python benchmarks/persist_demo.py resume --model <hf-model> --session demo
#   -> resumes with zero re-prefill, byte-identical output
```

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