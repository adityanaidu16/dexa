# Dexa Results

## Update (2026-06-28): validated on Llama-3.1-8B — where AM actually wins

Running on a real 8B model at 8000-token context exposed, then resolved, the
core question. Honest summary of the arc:

1. **Pure-importance key selection collapses at long context.** Attention
   Matching was healthy at <=3000 tokens but fell to ~0.04 needle-recall at 8000
   (flat across ratios), while H2O/SnapKV stayed ~0.95. Cause: a rare far-back
   needle gets ~0 importance under reference queries; mass-based selection keeps
   it. Fix: **mass-aware hybrid selection** (`mass_frac`/`recent_frac`).
2. **At moderate compression (16-128x) all good methods tie** (~0.95) — the
   single-needle task saturates; no method "wins" and AM is *not* better here.
3. **At extreme compression (>=256x) AM wins, with a widening gap.** With
   `mass_frac=1.0` (AM selects H2O's *exact* keys, so the only difference is
   AM's bias+value synthesis):

   | ratio | AM | H2O | SnapKV | AM-H2O | AM paired-win |
   |------:|---:|----:|-------:|-------:|--------------:|
   |  128x | 0.96 | 0.90 | 0.90 | +0.05 | 4/4 |
   |  256x | 0.94 | 0.89 | 0.89 | +0.05 | 4/4 |
   |  512x | 0.95 | 0.86 | 0.87 | +0.09 | 4/4 |
   | 1024x | 0.96 | 0.81 | 0.78 | +0.16 | 4/4 |

   AM holds ~0.95 while selection declines to ~0.80; at 1024x an 8000-token
   context is held in ~8 compact tokens. The gap is **purely AM's value
   synthesis** (same keys as H2O). This is the defensible "better context
   management" result — **AM's edge is at extreme compression, not moderate.**

Caveats (to make it publication-grade): single model, single-needle, 8k context,
4 seeds. Broaden to multi-needle, multiple models, longer contexts, more seeds.
AM also costs ~5x H2O (self-study refs + NNLS + lstsq) — justified at extreme
ratios, not at moderate ones. Reproduce: `dexa run --config configs/extreme-8b.yaml`.

The original (small-model) sections below are kept for the record; the 8B run
above supersedes their headline.

---

This document reports what the benchmarks actually show, including the places
where an honest reading contradicts the optimistic one. Three result sets are
relevant:

1. the **toy-backend reconstruction frontier** (`benchmarks/out/results.json`,
   `FakeBackend`),
2. the **real-model needle recall** on SmolLM2-360M
   (`benchmarks/out/niah_real.json`, 8 seeds), and
3. the **long-horizon agentic** result on SmolLM2-360M
   (`benchmarks/out/agentic.json`),

plus the **LMCache reuse-vs-compaction** framing
(`bench/lmcache_baseline.py`). What still needs the GPU cluster is marked
explicitly at the end.

A note on metrics. `recon_cosine` is cosine similarity between the compact and
full attention outputs (1.0 = perfect, higher = better). `recall_frac` in the
real-model runs is an affine rescaling of a summed gold-answer log-prob,
`(lp − floor)/(ceiling − floor)`, where `floor` is no-context and `ceiling` is
full-KV; it is **unbounded** and can exceed 1.0 (a denoising artifact — the
compact cache scores the gold answer slightly higher than the full cache because
it dropped distractor logit noise). Values near or above 1.0 mean the differences
being averaged are inside the model's own logit noise, not a capability gap.

---

## 1. Toy-backend reconstruction frontier (FakeBackend)

On the torch-free `FakeBackend`, Attention Matching dominates the reconstruction
frontier decisively. Mean `recon_cosine` over the NIAH/QA/multihop tasks, by
compression ratio:

| ratio | attention_matching | recent_window | random_subset |
|------:|-------------------:|--------------:|--------------:|
|    2x |             0.9999 |        0.9285 |        0.9495 |
|    4x |             0.9989 |        0.8275 |        0.8692 |
|    8x |             0.9934 |        0.7484 |        0.7692 |
|   16x |             0.9903 |        0.6091 |        0.6700 |

(`full_kv` is the 1.0 reference.) AM holds `recon_cosine > 0.99` out to 16x while
the selection baselines fall to ~0.6–0.7. Averaged over all >1x ratios: AM 0.996,
random 0.815, recent 0.778.

**This result overstates AM's real-world advantage and should be read as a
sanity check, not evidence.** `FakeBackend` uses deterministic linear-ish numpy
attention, which is close to the exact setting AM's least-squares fit assumes
(reproducing `softmax(QKᵀ)V` from a subset of keys with a per-key bias and a
linear value remap). AM is essentially solving the problem the toy model is
defined by, so near-perfect reconstruction is expected. The toy frontier
confirms the math is correct and the harness is fair; it does **not** establish
that AM wins on a real model. For that, see §2.

---

## 2. Real-model needle recall — SmolLM2-360M (8 seeds)

`benchmarks/niah_real.py` on `HuggingFaceTB/SmolLM2-360M-Instruct`, single-needle
recall, 8 seeds, `T ≈ 816`. Mean `recall_frac` by method and ratio:

| ratio | attn_matching | heavy_hitter | snapkv | recent_window | random |
|------:|--------------:|-------------:|-------:|--------------:|-------:|
|    4x |         0.970 |        0.979 |  0.944 |        −1.243 | −0.013 |
|    8x |         0.977 |        0.980 |  0.947 |        −1.170 | −0.903 |
|   16x |         0.996 |        0.974 |  0.960 |        −1.162 | −1.631 |
|   32x |         0.963 |        0.977 |  0.950 |        −1.358 | −2.387 |
|   64x |         0.888 |        0.807 |  0.906 |        −0.868 | −2.052 |
|  128x |         0.902 |        0.426 |  0.615 |        −0.468 | −1.885 |

### What the data supports

**The decisive, statistically significant AM win is at extreme compression
(128x).** Paired (per-seed) AM−HH differences:

| ratio | mean Δ (AM−HH) | std | p (paired t) | AM wins |
|------:|---------------:|----:|-------------:|--------:|
|    4x |         −0.009 | 0.077 |       0.764 |     6/8 |
|    8x |         −0.003 | 0.093 |       0.938 |     6/8 |
|   16x |         +0.022 | 0.046 |       0.252 |     5/8 |
|   32x |         −0.014 | 0.078 |       0.659 |     3/8 |
|   64x |         +0.081 | 0.278 |       0.466 |     5/8 |
|  128x |         **+0.476** | 0.217 | **0.001** | **8/8** |

At 128x AM keeps the needle (0.902) while H2O/HeavyHitter collapses (0.426) and
SnapKV degrades (0.615); the effect is large (Δ ≈ +0.48), unanimous (8/8 seeds),
and significant (p = 0.001). At 64x AM is ahead in the mean (0.888 vs 0.807) but
the per-seed difference is not significant (p = 0.47, high variance). The honest
headline is: **AM ≥ H2O, and AM degrades far more gracefully than H2O at extreme
compression (≥64–128x), where H2O collapses.**

### Honest caveats

- **The mid-ratio gaps (4–32x) are near or below significance.** Pooled over
  4–32x, AM (0.977) and HH (0.978) are a dead tie (p ≈ 0.94). The metric
  saturates here: every reasonable method lands at ~0.95–1.0 and many cells have
  `recall_frac > 1.0` (the denoising artifact above), so there is no headroom to
  resolve a difference. A claim that AM "beats H2O at 4–32x" is **not** supported
  by this data and at some configs is contradicted — independent sweep/ablation
  runs show HH beating AM at 16x, 32x, and 64x. AM is also higher-variance: on
  one seed it underperformed plain H2O even at 4–8x.
- **`recall_frac > 1.0` is a denoising caveat, not super-human recall.** Compact
  caches that drop distractor tokens can score the gold answer marginally higher
  than full-KV; this is logit noise, not capability.
- **The reference queries can peek at the answer.** The `self_study` reference
  queries are greedy continuations of the *full* context (which contains the
  needle), so generated reference positions can attend to the answer value. This
  inflates **all** methods' absolute recall equally (it does not bias AM vs HH —
  both get identical refs), but it makes the benchmark easier than a deployment
  where you cannot peek at the answer. The "preserves the needle at 128x" framing
  is therefore optimistic in absolute terms.
- **AM is not compute-matched.** Mean compaction time: AM 1.29s vs HH 0.25s
  (~5x), plus AM relies on the expensive self-study reference generation and
  per-head NNLS + ridge solves, whereas H2O is a single softmax+sum. Across
  4–32x you pay ~5x compaction compute for no measurable quality gain; the win at
  128x is what justifies the cost. A quality-per-second frontier should accompany
  any claim.
- **AM's true footprint is slightly worse than its token ratio.** AM additionally
  stores a per-key bias `beta` and a refit dense `Cv`, whereas selection-only H2O
  reuses original `K/V` with `beta = 0`. The overhead is small (~1/(2·head_dim))
  but real.
- **8 seeds is not enough to resolve the small gaps.** Per-ratio SE of the AM−HH
  difference is ~0.017–0.035 against gaps of ~0.003–0.022 at 4–32x; only the
  ~0.48 effect at 128x is resolvable at n=8. More seeds would most likely confirm
  the mid-ratio tie, not surface a win.

---

## 3. Long-horizon agentic — SmolLM2-360M

`benchmarks/agentic_real.py` on SmolLM2-360M: a multi-turn trajectory that plants
facts early and queries them late, comparing memory strategies on **late
recall**, **peak memory**, and **compute**. Summary:

| strategy | late_recall | peak_tokens | peak_kv (MB) | compactions | compute (s) |
|---|---:|---:|---:|---:|---:|
| full_kv | 1.000 | 1252 | 102.6 | 0 | 0.0 |
| truncate_recent | 0.131 | 300 | 24.6 | 0 | 0.0 |
| dexa:attention_matching | 0.156 | 465 | 38.1 | 7 | 39.7 |
| dexa:heavy_hitter | 0.225 | 465 | 38.1 | 7 | 30.9 |

The story:

- **Full-KV is the recall ceiling but is unbounded.** It recalls every planted
  fact (1.0) but its KV grows with the trajectory (1252 tokens, 102.6 MB, and
  climbing) — the memory wall.
- **Dexa's bounded working memory beats truncation on late recall at bounded
  cost.** Both Dexa strategies cap peak memory (465 tokens / 38 MB, set by the
  working-memory budget) and both beat hard recent-window truncation on
  late-recall (AM 0.156 and HH 0.225 vs truncation 0.131). Truncation is free but
  forgets early facts; Dexa pays a transient recompute cost (7 compactions, ~31–40
  GPU-seconds) to retain a compressed summary of the whole history within a fixed
  footprint.
- **In this agentic setting H2O-style heavy_hitter outperforms Attention
  Matching on late recall** (0.225 vs 0.156) and is cheaper (30.9s vs 39.7s).
  This is consistent with §2: at the moderate per-compaction ratios used here AM
  carries no quality advantage over H2O, and its extra machinery costs more
  compute. AM's edge is at extreme compression, which this trajectory does not
  reach.
- **Absolute late-recall is low for every bounded method** (0.13–0.23). The task
  is hard (facts planted up to 8 turns before the query), the model is small
  (360M), and the budget is tight; the meaningful comparison is *relative* —
  bounded Dexa > truncation at equal/near-equal memory — not the absolute number.

Peak memory for the Dexa rows is the *maintained* post-compaction working set;
the transient prefix re-prefill spike is reported separately
(`peak_recompute_tokens` / `transient_recompute_tokens`) and is not counted as
retained memory, matching `WorkingMemory`'s own accounting.

---

## 4. LMCache reuse vs. Dexa compaction

These are the two answers to the memory wall, and the framing matters more than
the absolute bytes. `bench/lmcache_baseline.py` runs them on the same request
stream over a shared `TieredCacheStore`:

- **Reuse + tiering (LMCache-style)** never recomputes a seen prefix and spills
  KV down GPU→CPU→NVMe instead of discarding it. It **saves compute** (prefix
  hits avoid re-prefill) but **does not bound memory**: retained KV grows with the
  total *unique* context the system has ever handled.
- **Compaction (Dexa)** shrinks old KV into a fixed working set, so memory stays
  flat regardless of how much unique context streams through.

On a shared growing transcript, the LMCache footprint climbs from ~22 KB to
~196 KB at a 0.74 prefix-reuse hit rate, while the Dexa `WorkingMemory` flattens
at its ~32 KB budget after 7 compactions — LMCache grows monotonically, Dexa
stays ≤ budget. These numbers come from the `FakeBackend` reuse harness with
exact reuse/recompute *counts*; the absolute bytes and the recompute-avoided
GPU-seconds are **model-relative** (computed via `CostModel` defaults), and the
NVMe tier is simulated via a modeled access latency rather than real disk I/O.
The two approaches are complementary, not mutually exclusive: reuse avoids
recompute, compaction bounds memory, and Dexa's `CompactCache` + tiered store can
host both.

---

## 5. What still needs the GPU cluster

The following are built and structurally tested but were **not** run here (CPU is
reserved; large/vLLM/GPU paths are out of scope for this machine):

- **vLLM real run.** `engine/vllm_backend.py` implements the cluster-grade
  backend. Its structural surface is tested everywhere, but the post-RoPE q/k/v
  hook extraction, the KV-connector path (TP>1 / disaggregated), and the
  beta-aware compact-decode shim only execute on a GPU node with vLLM installed;
  they are marked `pragma: no cover` and `generate`/`score` against a compact
  cache raise a clear `RuntimeError` until a site finishes wiring the
  deployment-specific beta-aware attention backend. The faithful CPU eval path
  remains `HFBackend`.
- **Big-model Attention Matching.** All AM quality numbers above are either toy
  (`FakeBackend`, §1) or on SmolLM2-360M (§2–3). Whether AM's extreme-compression
  advantage holds, widens, or shifts on an 8B-class model with longer contexts is
  unverified.
- **STILL full training.** `still/train.py` was only smoke-tested on
  `hf-internal-testing/tiny-random-LlamaForCausalLM` (KL decreases over ~12
  steps; the tiny model's near-uniform logits make the absolute KL ~5e-5, so the
  test asserts the loss *drops*, not that it hits a quality bar). The
  identity-init reconstruction at `t == T` is verified to ~1e-7. Real distillation
  quality (long contexts, SmolLM2/8B, many documents) and any head-to-head of
  STILL's amortized forward pass against AM's per-context fit require the cluster.
- **More seeds / compute-matched frontier for §2.** Resolving the 4–32x
  tie-or-not and producing a quality-per-GPU-second curve needs more seeds and a
  compute-matched run.

---

## Reproducing

All CPU-runnable results use `FakeBackend` or the tiny model and finish in
seconds. The real-model SmolLM2 benchmarks (`benchmarks/niah_real.py`,
`benchmarks/agentic_real.py`) must be run with `HF_HUB_OFFLINE=1`. Raw outputs
live in `benchmarks/out/` (`results.json`, `niah_real.json`, `agentic.json`) with
plots (`frontier.png`, `niah_frontier.png`, `agentic_tradeoff.png`,
`memory_saving.png`). The full test suite is green (62 passed, 3 skipped — the
skips are vLLM-absent functional tests).
