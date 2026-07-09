# Dexa

**Persistent, portable inference state for open-model agents.** Keep an agent's KV cache as a server-side object, resume it on *any* GPU — after a restart, a replica move, or a spot preemption — with **identical output and zero re-prefill**, and compact the cold tail so long sessions stay cheap. Built for open models on vLLM: the statefulness frontier providers give you for free, and self-hosters don't have.

> **Status (v0.1, honest).** **Portable, lossless resume is validated end-to-end on a real 8B (Llama-3.1-8B, A100)** and is now **faster than cold re-prefill on a warm GPU**: save a session, drop the cache, reload it in a fresh process, and continue with **zero re-prefill and bit-identical output** — **3.7–5.5× faster than re-prefilling at 8k–64k** (within-run; ~4.3× at 64k), surviving a restart, a replica move, or a spot preemption ([`docs/RESULTS.md`](docs/RESULTS.md), 2026-07-09).
> An honest note on the arc: an initial 8B/GPU run had resume *losing* to re-prefill because the bf16 KV load widened to fp32 on the host (~10× below hardware); keeping bf16 end-to-end (mmap → device `bfloat16`, no host widen) dropped the 64k load from ~25 s to ~14 ms and flipped it. The old "14–25× faster" figure was CPU/SmolLM2 and did not transfer to GPU — the numbers above are the real 8B/A100 measurement. Compacted state (fewer bytes still) is the next lever. Try the demo in 2 commands below.
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
- **Native-precision state** — persist KV at the model's actual dtype (`bfloat16`/`float16`), not upcast to fp32. Lossless for a genuinely bf16/fp16 model and **~2× smaller** persisted state (and ~2× less bytes to move across a replica/tier). Automatic via `spec.dtype`; `precision=` overrides.
- **Memory-mapped blob format** (`SessionStore(..., format="blob")`) — drops the `.npz` ZIP container for a header + raw KV bytes loaded via `mmap`, so a resume is a **zero-copy view** (fp32) paged in during the host→device copy instead of a full up-front decode. **~1.7× faster resume load** vs `.npz` (`benchmarks/persist_format_bench.py`). An optional Rust codec (`native/kvcodec`) parallelizes the save-side pack; the on-disk format is identical with or without it.
  - **bf16 fast path (the common case):** `load(..., keep_native=True)` keeps bf16 state as its raw uint16 bits (zero-copy mmap, no host fp32 widen); `HFBackend` reinterprets them straight to a device `bfloat16` tensor via `torch.frombuffer`. This dropped the 64k resume load from ~25 s to ~14 ms on the 2026-07-09 A100 run and flipped resume to **3.7–5.5× faster** than re-prefill (bit-identical) — see [`docs/RESULTS.md`](docs/RESULTS.md). The persist bench uses it by default (`--no-native-load` restores the old fp32 widen for A/B).

### Mutable state with incremental recompute
- Segment-level dependency tracking: the cache is structured so the system knows which downstream segments depend on which upstream ones.
- On a mid-context edit, recompute only the affected segment(s) and everything causally downstream — not the entire prefix.
- The thing nothing in open-source does today.

### Versioning & branching  *(prototyped: `dexa.segment.SegmentedSession`)*
- `commit` a turn and version the result; `branch` a session for sub-agents or speculative paths; `diff` and `rollback` — all on the live segmented context, with mutations updating KV via incremental recompute and snapshots/forks copying KV instead of re-prefilling.
- Multi-agent systems fork shared context without duplicating full prefills (`branch()` copies the KV; the shared prefix is never re-encoded).
- Validated end-to-end (`tests/test_segmented_session.py`): mutations stay behaviorally identical to a full re-prefill, branches are isolated, rollback restores, and a session persists + reattaches via the blob store.

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

### Phase 1 — Mutable state + incremental recompute  *(substrate prototyped)*
- Segment-level dependency tracking (causal DAG over context segments) — `dexa.segment`: content-identified `Segment`s, a `RecomputePlan`, and `plan_incremental(prev, new)` (pure, unit-tested).
- **Exact incremental recompute on mid-context edits** — `HFBackend.recompute_incremental` reuses the unchanged segment prefix and re-encodes only from the first change onward. Validated on a real model (tiny-random Llama, CPU) to be *numerically equivalent to a full re-prefill with an identical greedy continuation, and the reused prefix bit-identical to the cached state* (`tests/test_incremental_recompute.py`).
- Handle file/tool-result edits, replayed-message divergence, header churn.
- **Gate (measured):** on a simulated agent loop (`benchmarks/incremental_recompute_bench.py`), **4.6× fewer tokens reprocessed** per mutating turn vs full re-prefill with a small stable prefix — the reduction grows toward order-of-magnitude as the stable prefix dominates (the real long-horizon case). Honest limit: an edit *early* in a long context still forces recompute of everything after it; the win concentrates on appends and edits near the tail.
- **Selective recompute (CacheBlend, EuroSys'25) — prototyped.** For the stale-but-content-identical suffix after an edit, reuse its KV and recompute only the top-fraction highest-deviation ("HKVD") tokens (`HFBackend.recompute_selective`, `dexa.segment.selective`). Validated (`benchmarks/selective_recompute_bench.py`): on a length-preserving mid-context edit, **HKVD selection removes downstream attention-output error faster than recency or random at every recompute level** (recency is *worst* — stale tokens aren't at the tail). The *ordering* is confirmed on random weights; the *magnitude* (CacheBlend's "~15% recovers most") needs a trained model + GPU, where the deviation distribution is far peakier. This is what beats prefix caching's reuse extent; Dexa's differentiation over LMCache's CacheBlend is applying it to the *mutation* case on a versioned segment graph.
- **Exact RoPE re-phasing — built.** Reused segments after a *length-changing* edit are shifted in position; since keys are post-RoPE and RoPE composes, their keys are re-phased by the position delta **exactly** with no forward pass (`selective.rope_rephase_keys` + `HFBackend.rephase_cos_sin`; values carry no RoPE and are reused as-is). Gated by a position-only exactness test — the same tokens prefilled at `offset=0` vs `offset=delta` differ only by RoPE phase, and re-phasing reconstructs them (`tests/test_rope_rephase.py`); the generalized selective path reproduces a full re-prefill at full recompute (`tests/test_selective_engine.py`). This makes `recompute_selective` handle length-changing edits. (Re-phasing corrects the *position* component exactly; its standalone error-reduction is largest when content is stable and position shifts, e.g. RAG-chunk reuse, and small when content change dominates — the content residual is HKVD's job.)
- **Next:** the compute realization (forward only the selected tokens, layer-by-layer sparse — turns the token-count saving into wall-time), then Phase 1.x versioning/branching (`commit`/`branch`/`diff`/`rollback`) on the segment graph.

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