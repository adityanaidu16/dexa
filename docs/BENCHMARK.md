# The target: accuracy at fixed KV-memory (the frontier win)

One falsifiable claim. Everything is built to prove or kill it.

## Claim

> On long-context QA, a **cartridge** (trained compact KV) sits on a strictly
> better **accuracy-vs-KV-memory** frontier than every baseline: as accurate as
> full-context at a fraction of the memory, and more accurate than RAG and than
> training-free KV compression at the same memory budget.

## The plot (the whole result)

X = KV memory held for the corpus (bytes, or compression ratio T/t).
Y = task accuracy.
One point per (method, memory budget). **Win = the cartridge curve is on the
Pareto frontier and dominates at ≥1 memory budget.**

## Methods (both baseline families on one plot)

KV-mechanism baselines (same level as a cartridge):
- `full_context` — prefill the whole corpus (accuracy ceiling, max memory)
- `h2o` (HeavyHitter), `snapkv` — training-free KV compression (drop tokens)
- `attention_matching` — our analytic compactor (training-free, our earlier win)

Task baselines (different level, same job):
- `rag` — top-k retrieval into the prompt (the incumbent for corpus QA)

The method under test:
- `cartridge` — trained compact KV (per corpus, offline)

All non-full methods evaluated at matched memory budgets (e.g. 4×, 16×, 50×,
128× compression) so the X-axis is comparable.

## Datasets

- **LongBench** single-doc QA subsets (NarrativeQA, Qasper, MultiFieldQA) —
  real, standard, F1-scored, ~10–20k-token contexts. The credibility anchor.
- **RULER** (NIAH variants, variable tracking, aggregation) — controlled
  synthetic for a clean compression sweep and to locate where methods break.

## Metric

Task-appropriate accuracy (LongBench: token-F1 / substring-EM per task; RULER:
exact-match / recall). Memory: bytes of KV held for the corpus (compact tokens ×
per-token KV bytes), reported also as compression ratio.

## Success threshold (pre-registered, so we can't move the goalposts)

At a fixed compression in **[16×, 50×]**:
- `cartridge` F1 ≥ `full_context` − **2 points** (matches the ceiling), AND
- `cartridge` F1 ≥ best training-free KV method (`h2o`/`snapkv`/`attention_matching`) + **5 points**, AND
- `cartridge` F1 ≥ `rag` + **3 points**.

If all three hold on LongBench single-doc QA with an open 7–8B model, the frontier
claim is proven. If not, it's killed (and we report honestly which baseline wins).

## Honest risks (the reasons this could fail)

1. **Cartridge quality reproduction** — the crux; failed at small scale (CPU/360M)
   due to weak self-study data. Needs a real model + corpus-conditioned synthetic
   Q&A + GPU training. This benchmark is how we find out.
2. **RAG is strong** on single-hop extractive QA — beating it by 3 F1 is not free.
3. **Per-corpus training cost** — ignored on this (accuracy-vs-memory) axis; it's
   the offline compile. Reported separately as break-even for the cost story.

## What runs where

The **harness** (datasets, methods, metrics, frontier plot) is CPU-built and
plumbing-validated with a tiny model. The **decisive run** (open 7–8B model +
cartridge training + real LongBench/RULER) needs one GPU box.
