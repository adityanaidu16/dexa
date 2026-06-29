# Dexa Cartridges — the context compiler

**Product:** turn a large *static* corpus (a repo, a doc set, a knowledge base)
into a small, portable, trained KV cache — a **cartridge** — that loads into a
standard inference engine (vLLM) as a precomputed prefix. Queries then run with
the corpus "in context" at a fraction of the cost and memory of stuffing the raw
corpus in every prompt, and **beyond the model's context window**.

This is the productization of *Cartridges* (Eyuboglu et al., 2025) on open models,
served on top of (not replacing) vLLM.

```
  cartridge compile ./my-corpus  ->  my-corpus.cartridge     # offline, the IP
  vllm serve ... + dexa sidecar loads my-corpus.cartridge    # thin KV injection
  query("...") attends over the cartridge like real context  # cheap, small, portable
```

## Why a cartridge beats the obvious alternatives

| approach | cost/query | memory | quality | > context window |
|---|---|---|---|---|
| Full-context (ICL) + prompt caching | high | **huge** (full KV) | ceiling | no |
| RAG (top-k retrieval) | low | low | lossy (retrieval misses) | yes |
| **Cartridge** | low | **tiny** (~50-100x) | **~ceiling** | **yes** |

The decisive edges over *prompt-cached full context* (the real competitor):
- **Memory ~50-100x smaller** → far more corpora per GPU; hold corpora that don't
  fit in KV at all.
- **Beyond the context window** → compress a 1M-token corpus for a 128k model.
- **Portable artifact** → version / share / swap; not pinned to one warm instance.

The catch (be honest in the benchmark): a cartridge costs **offline training** per
corpus (minutes-hours). It wins only for **high-reuse static corpora** — report
the **break-even** (after K queries, cheaper than re-prefill / prompt-cache).

## Architecture

A cartridge is a **trained compact KV cache with no attention bias** — so it
serves as an ordinary small KV prefix (no custom attention kernel needed, unlike
Attention Matching). It reuses `CompactCache` (biases = 0).

- `dexa.cartridge.artifact.Cartridge` — the portable artifact (per-layer compact
  K/V + positions + metadata) with `save()` / `load()`.
- `dexa.cartridge.selfstudy` — generate self-study training data from the corpus
  (synthetic questions; teacher = model with the *full* corpus in context).
- `dexa.cartridge.compiler.CartridgeCompiler` — trains the cartridge K/V by
  gradient descent through the **frozen** model: minimize KL(teacher || student)
  on answer tokens, where teacher = full-corpus context, student = cartridge.
  Initialized from a downsample of the corpus KV (warm start).
- `dexa.engine.vllm_cartridge` — load a cartridge into vLLM as a KV prefix
  (import-guarded; runs on the cluster).
- `dexa.bench.corpus` — the GTM benchmark: corpus-QA quality + cost + memory +
  break-even for full-context vs RAG vs cartridge.

## Compiler (the method)

1. **Layout.** Cartridge has `t` compact tokens at absolute positions
   `linspace(0, T-1, t)` (T = corpus length) so appended query tokens get correct
   RoPE phases and the cartridge spans the corpus.
2. **Warm start.** Prefill the corpus once; initialize the cartridge K/V from the
   corpus KV at the layout positions (a downsample). Already a decent cache.
3. **Self-study.** Generate `n` synthetic questions about the corpus; for each,
   get the teacher next-token distribution with the *full* corpus in context.
4. **Train.** Make the cartridge K/V `requires_grad`; freeze the model. For each
   self-study item, forward the question with the cartridge as `past_key_values`,
   compute KL(teacher || student) on the answer span, backprop into K/V, Adam.
   Everything else (model weights) is frozen.
5. **Emit** the trained `Cartridge` artifact.

## Novelty / moat ladder

1. **v1 (ship first):** faithful Cartridges on open models + the portable-artifact
   format + vLLM serving + the public benchmark. (Systems + productization.)
2. **v2 (the research wedge):** *amortize* the per-corpus training — a trained
   encoder (STILL-style perceiver) that compiles any corpus in a forward pass,
   killing the minutes-hours cost. This is the hard-to-copy contribution.
3. **hybrid:** warm-start with Attention Matching's analytic solve to cut training
   steps.

## GTM

Public corpus + QA (large codebase + repo-QA, or HELMET/LongBench long-doc).
Headline: *"cartridge matches full-context quality at RAG cost, ~100x less
memory, beyond the context window — open model, reproduce the repo."* Quality
parity is make-or-break; prove it first.

## Status (v0.1 — honest)

**Built and tested (82 passing):** the `Cartridge` artifact (+ save/load), the
`CartridgeCompiler` training loop (KL-distillation through the frozen model —
grad flows, KL converges to ~0), the vLLM serving sidecar (zero-bias = plain
prefix injection, no custom kernel), and the three-way GTM benchmark
(full-context vs RAG vs cartridge with break-even) + launch script.

**NOT yet reproduced: cartridge QA quality.** On a 360M model on CPU the trained
cartridge does **not** beat no-context on held-out QA — and we learned exactly
why, which is the crux of the whole method:

- **The self-study data distribution is everything.** Distilling generic prompts
  ("summarize the key facts") fails — the teacher's answers don't contain the
  specific facts, so the cartridge never learns them. Distilling the corpus's own
  spans (corpus-LM) **actively hurts QA**: it teaches "continue this corpus text,"
  which misleads the model when the input is a *question*. This is precisely why
  Cartridges uses **corpus-conditioned synthetic Q&A** self-study.
- Generating good fact-bearing Q&A needs a **capable model + sampling** — a 360M
  model with greedy decoding can't manufacture it. So quality reproduction is a
  **GPU + real-model** experiment, not a CPU one. (This matches the paper: minutes
  -to-hours of per-corpus self-study on real models.)

**Next experiment (the make-or-break):** on GPU with a real model, generate
corpus-conditioned synthetic Q&A self-study and re-run the benchmark; the quality
column is what the entire product rests on. The pipeline is ready for it; only the
self-study data generator needs the capable model.
