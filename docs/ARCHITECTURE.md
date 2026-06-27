# Dexa Architecture

Dexa is a **compaction-first inference-state engine**. It is organized around a
single idea: a transformer's KV cache for a chunk of context can be *compacted* —
replaced by a much smaller cache (compact keys, values, and per-key attention
biases) that preserves the model's behavior when future tokens attend to that
chunk. Dexa owns the lifecycle of that compact state: producing it, persisting
it, tiering it, and stitching it back into an engine for decode.

This document describes the layering, the data model, the exact compact-decode
path, and each compactor and its relationship to the literature. It is
descriptive of what is built today; see `docs/RESULTS.md` for what the
benchmarks actually show (and what still requires a GPU cluster).

---

## 1. Layering

```
                 +------------------------------------------+
   bench         |  bench/  +  benchmarks/                  |   harness, tasks,
   (CPU)         |  runner, metrics, report, agentic,       |   cost model,
                 |  lmcache_baseline                        |   plots
                 +---------------------+--------------------+
                                       |
   working        +--------------------v--------------------+
   memory         |  memory/                                |   bounded long-horizon
                  |  WorkingMemory  ·  TieredCacheStore     |   memory + reuse/tiering
                  +--------------------+--------------------+
                                       |
   compactors     +--------------------v--------------------+
   (numpy/torch)  |  compaction/                            |   KVCache -> CompactCache
                  |  AttentionMatching · baselines · STILL  |
                  +--------------------+--------------------+
                                       |
   backends       +--------------------v--------------------+
   (torch/vllm    |  engine/                                |   text <-> KV, refs,
    at the edge)  |  ModelBackend: Fake · HF · vLLM         |   decode/score
                  +--------------------+--------------------+
                                       |
   core           +--------------------v--------------------+
   (numpy only)   |  core/types.py                          |   data model + contracts
                  +------------------------------------------+
```

The boundaries are contracts, not suggestions. The files that define them —
`core/types.py`, `engine/base.py`, `engine/fake.py`, `compaction/base.py` — are
frozen; everything else is built against them.

- **core** (`dexa.core.types`) — the data model and the two key protocols
  (`CacheStore`, the `ModelSpec`/`CostModel` value types). Pure numpy, no torch.
- **engine/backends** (`dexa.engine`) — a `ModelBackend` turns text into a
  `KVCache`, produces reference queries, and decodes/scores against a full or a
  compact cache. Three implementations exist: `FakeBackend` (deterministic numpy
  attention, no torch — the CI/quality workhorse), `HFBackend` (a real
  transformer via `transformers`, eager-attention monkeypatch for differentiable
  per-key beta injection), and `VLLMBackend` (the cluster-grade counterpart;
  structural surface complete, the post-RoPE q/k/v hook extraction and the
  beta-aware compact-decode shim are written but only runnable on a GPU node with
  vLLM installed).
- **compactors** (`dexa.compaction`) — a `Compactor` maps a `KVCache` (+ optional
  `RefQueries`) to a `CompactCache` under a `CompactionBudget`. Attention
  Matching is the flagship; the selection baselines and STILL implement the same
  interface so the benchmark is apples-to-apples.
- **working memory** (`dexa.memory`) — `WorkingMemory` is the bounded,
  iteratively-compacted memory for long-horizon agents; `TieredCacheStore` is the
  GPU→CPU→NVMe persistence/reuse axis that implements the `CacheStore` protocol.
- **bench** (`dexa.bench` + `benchmarks/`) — the reconstruction/recall harness,
  the agentic long-horizon harness, the LMCache reuse baseline, the cost model,
  and plotting.

---

## 2. The data model

All tensors at the core and compaction boundary are **numpy float32**. Torch
lives only inside the HF/vLLM backends and the STILL subpackage, which convert at
their edges. This is what keeps the compaction math, the baselines, and the
entire benchmark harness CPU-testable with no GPU or torch dependency.

Per-layer shapes (from `core/types.py`):

| Tensor | Shape | Notes |
|---|---|---|
| full keys / values `K`, `V` | `[n_kv_heads, T, head_dim]` | `K` is **post-RoPE** |
| reference queries `Q` | `[n_q_heads, n_ref, head_dim]` | post-RoPE |
| compact keys / values `Ck`, `Cv` | `[n_kv_heads, t, head_dim]` | `t << T` |
| compact biases `beta` | `[n_kv_heads, t]` | per-key additive logit bias |

Under grouped-query attention (GQA) several query heads map to one kv-head; the
mapping is `q_head // (n_q_heads // n_kv_heads)` (`ModelSpec.kv_head_of`).

### KVCache (the full cache)

`KVCache` holds a `ModelSpec`, a list of per-layer `LayerKV(key, value)`, the
absolute `positions` the keys were computed at, and optional `token_ids`/`meta`.
It is what a backend's `prefill()` returns.

### CompactCache (the persistent Dexa state object)

`CompactCache` is the portable artifact Dexa produces and persists. Because the
budget may differ per kv-head (sensitivity allocation), each `CompactLayer`
stores per-head **ragged** lists: `keys[h] : [t_h, d]`, `values[h] : [t_h, d]`,
`biases[h] : [t_h]`, and `positions[h] : [t_h]` — the absolute positions each
compact key stands in for.

Two fields make the cache self-describing:

- `logical_length` — the original sequence length `T`. The physical size is
  `budget` (total compact tokens). New tokens appended after this cache must
  receive position ids starting at `logical_length` so RoPE phases stay correct.
- `compression_ratio` — `logical_length / (mean compact tokens per layer per
  kv-head)`. Higher = more compression.

### The numpy-float32 boundary

A backend converts torch→numpy when it returns a `KVCache` or `RefQueries`, and
numpy→torch when it decodes against a cache. A compactor only ever sees numpy.
This is a deliberate seam: every compactor and every baseline is exercised in CI
against `FakeBackend` with zero torch in the loop, and the same compactor object
runs unchanged against a real model.

### The exact compact-decode path

Decoding a new query `q` against a `CompactCache` is **not** ordinary attention.
Two things differ from a raw cache, both load-bearing:

1. **Per-key biases.** The compact scores are
   `softmax(q · Ck^T · scale + beta) · Cv`, i.e. the per-key bias `beta` is added
   **additively into the pre-softmax logits**, one scalar per compact key.
   `scale = 1/sqrt(head_dim)`. Selection-only baselines set `beta = 0` and reuse
   the original `K/V`; Attention Matching and STILL fit nonzero `beta`. The HF
   backend injects `beta` through its eager-attention monkeypatch (a 4D additive
   mask term); the vLLM backend's beta-aware attention shim is the cluster-only
   equivalent.
2. **Logical-length positions.** RoPE phase for the new token is taken from
   `CompactCache.logical_length` (= original `T`), **not** from the physical
   compact length `t`. Each compact key already carries the absolute `positions`
   it represents, so the relative phase between a new query and a compact key
   matches what the full cache would have produced. A cache that compacted 800
   tokens into 25 still decodes the next token as position 800.

This is why a `CompactCache` is composable with raw KV in a single fused decode:
the compact prefix occupies positions `[0, compact_end)`, a verbatim recent block
occupies `[compact_end, total)`, and the next query decodes at position `total`.

---

## 3. The compactors

All compactors share the `Compactor` interface (`compact(cache, budget, *,
ref_queries=None) -> CompactCache`) and a `name`. `needs_ref_queries` advertises
whether the method consumes reference queries (Attention Matching does; pure
selection baselines do not; STILL does not at inference time).

### 3.1 Attention Matching (flagship)

`compaction/attention_matching.py`. Implements *"Fast KV Compaction via Attention
Matching"* (Zweiger, Fu, Guo, Kim, 2026). The goal: replace a layer's `K, V`
`[T, d]` with `Ck, Cv` `[t, d]` plus per-key biases `beta` `[t]` so that, for a
set of **reference queries**, the locally-normalized attention output
`softmax(q · Ck^T · scale + beta) · Cv` reproduces the full-cache output
`softmax(q · K^T · scale) · V` as closely as possible. The fit is independent per
layer and per kv-head; under GQA the query heads sharing a kv-head are stacked so
one compact head serves the whole group. Three stages per kv-head:

1. **Key selection** — which original keys to keep. `highest_attention` (RMS
   attention-importance top-`t`, default) or `omp` (greedy orthogonal matching
   pursuit on the mass-matching residual).
2. **Bias fit** — non-negative least squares (`scipy.optimize.nnls`, with a
   never-raising fallback) so the kept keys reproduce the total attention *mass*
   each reference query placed on the full key set; `beta = log(w)`.
3. **Value fit** — least squares mapping the compact attention weights back to
   the full attention outputs to obtain `Cv`.

Budget allocation can be `uniform` or `sensitivity` (paper Algorithm 4):
concentrated-attention heads get fewer keys, diffuse heads get more, preserving
the per-layer total.

**Numerical hardening of the value fit.** The value-fit matrix
`X = softmax(Q · Ck^T · scale + beta)` becomes rank-deficient whenever a kept
compact key receives ~0 attention across every reference query (routine with
duplicate/collinear keys, where the NNLS bias fit zeroes all but one). A plain
`lstsq` then drives the unconstrained `Cv` rows to ~1e9–1e10 — harmless in
isolation (the key carries ~0 weight) but a float32 overflow once fused with raw
KV. The fit is hardened in three layers: (1) **dead-key drop** — keys with total
reference weight below a threshold are excluded and their `Cv` left at 0; (2)
**bounded solve** — live columns are solved via the ridge normal equations
`(XᵀX + λI) Cv = XᵀY` in float64, with a scale-adaptive `λ = 1e-6 · mean(diag
XᵀX)` even when `value_ridge == 0`, so the solve stays well-posed without
perceptibly altering a well-conditioned fit; (3) **safety clip** — dropped rows
are clamped into `V`'s empirical per-dim range. Net effect: `|Cv|` drops from
~1e10 to `V`'s scale (~1–7×) with no reconstruction regression. Caveat: only
dropped rows are hard-clipped; a pathological set of near-identical *live* keys
could still exceed `V`'s range — pass `value_ridge > 0` for those.

### 3.2 Selection baselines

`compaction/baselines.py`. Selection-only methods, each keeping a subset of the
original `K/V` with `beta = 0`, so Attention Matching is benchmarked
apples-to-apples on the same interface and budget:

- **FullKV** — no compaction; the quality upper bound.
- **RecentWindow** — keep the most recent tokens.
- **HeavyHitter** — H2O-style (Zhang et al., *H2O*, 2023): keep the keys with the
  highest accumulated attention mass plus a small recent window. This is the
  primary baseline AM is measured against.
- **SnapKVLite** — score keys by attention from only the most recent reference
  queries, plus a recent window (after *SnapKV*, Li et al., 2024).
- **RandomSubset** — deterministic random subset; the floor.

Selection scores aggregate the query heads sharing each kv-head under GQA.

### 3.3 STILL (amortized perceiver)

`compaction/still/`. The amortized counterpart to Attention Matching: a small
per-layer Perceiver, trained once against a frozen base model, maps a full KV
cache to a compact one in a **single forward pass** — no per-context numerical
fit. torch lives entirely inside this subpackage and is imported lazily so the
rest of dexa stays torch-free.

`StillPerceiver` (one per layer, parameters shared across kv-heads, kv-heads as
the batch axis) does, per forward: (1) un-rotate cached post-RoPE keys with the
model's *inverse* RoPE at their original positions; (2) form width-`2d` input
tokens `[K_unrot ; V]`; (3) `t` learnable latent queries cross-attend into them
using the perceiver's own internal RoPE (latents spread by `linspace(0, T-1, t)`);
(4) a multi-head latent self-attention block + MLP coordinate the latents; (5)
key/value/beta output heads project each latent (no final RMSNorm, so per-key
norm survives); (6) re-rotate the compact keys with the model's RoPE at the
latents' evenly-spaced absolute positions.

The module is **identity-initialized**: cross-attn value projection is identity,
the key/value heads are coordinate selectors reading the `K_unrot`/`V` halves,
q/k projections are zero-weight + constant-bias with a learnable routing
temperature (init 30) that makes routing essentially one-hot on RoPE phase, and
the self-attn/MLP output projections are zero (residual no-ops). At init each
latent copies its positionally-nearest input, so at `t == T` the compact cache
reproduces the input KV to ~1e-7 (verified). Training (`still/train.py`) is
KL-distillation: the frozen teacher decodes answer tokens over the full cache,
the student decodes the same tokens over the compact cache the perceivers
produce, and only the perceiver parameters get gradients. CPU-runnable on a tiny
model in seconds; full-scale distillation quality requires the cluster.

`StillCompactor` (`name="still"`, `needs_ref_queries=False`) converts
numpy→torch, runs the per-layer perceivers in one `no_grad` pass, and packs the
result into a `CompactCache`. It accepts trained perceivers or lazily builds
identity-init perceivers sized to `budget.target_t`, so the forward path is
always runnable and testable.

### Relationship to the literature

- **Attention Matching** (Zweiger, Fu, Guo, Kim, 2026) — the per-context
  numerical method Dexa implements as its flagship: fit compact `Ck/Cv/beta` to
  reference queries by matching attention output. Dexa adds the sensitivity
  allocation (Algorithm 4) and the value-fit hardening above.
- **Cartridges** (Eyuboglu et al., 2025) — trains a small persistent KV
  "cartridge" per corpus, amortizing context cost across many queries. STILL sits
  in the same amortized family but compacts an *arbitrary* existing KV cache in
  one forward pass rather than learning a fixed per-corpus artifact, and Dexa's
  `CompactCache` + `TieredCacheStore` are exactly the persistence substrate such
  cartridges need.
- **STILL** — the amortized-perceiver compactor: instead of paying AM's
  per-context fit, learn the compaction map once and apply it in a forward pass.
  This is the speed/quality trade against AM that the cluster is meant to settle.

The two competing answers to the long-context **memory wall** are made explicit
in `memory/`: **reuse + tiering** (LMCache-style: never recompute, spill KV down
a tier — saves compute but does **not** bound memory) versus **compaction**
(Dexa's `WorkingMemory`: shrink old KV into a fixed working set — memory stays
flat). `bench/lmcache_baseline.py` runs them head to head; see `docs/RESULTS.md`.

---

## 4. Working memory and persistence

`WorkingMemory` keeps a bounded combined cache `[ compact_memory ; recent_raw ]`:
the most recent `keep_recent_tokens` are kept verbatim, everything older is
compacted to a fixed budget. As the trajectory grows the *same* fixed compact
budget holds an ever-larger logical span. Because a KV cache is not composable
from independently-prefilled chunks, the compact memory is rebuilt from a correct
**prefix prefill** (causality: prefilling `[0, m)` yields exactly the same KV for
those tokens as prefilling the whole sequence). The maintained working set is
bounded; compaction pays a transient **recompute** cost (re-prefilling the older
prefix to recompress it), surfaced honestly as `peak_recompute_tokens`.

`TieredCacheStore` implements the `CacheStore` protocol over a configurable
GPU→CPU→NVMe hierarchy. New entries land in the top tier; over-capacity tiers
demote their LRU entry down a tier; an entry pushed off the bottom tier is
dropped (forcing later recompute). `get()` and a deduplicating `put()` promote
hot entries back to GPU; a content-hash index backs `has()` for prefix reuse. It
is numpy/in-memory and torch-free; the NVMe tier is *simulated* via a modeled
access latency (`fixed_latency_s + nbytes/bandwidth`) rather than on-disk `.npz`.
`stats()` reports per-tier used/peak/capacity/count, reuse hit rate, demotions,
dropped evictions, and total modeled access seconds. The tier
bandwidths/capacities are illustrative orders of magnitude, not measured.
