# Dexa benchmark harness

Proves the Dexa thesis: a small **compact cache** can replace a big KV cache
while preserving model behavior. The harness sweeps a matrix of
**compactor × compression-ratio × task**, measures quality and system cost, and
draws the frontier (quality vs compression). It is **backend-agnostic**: the
exact same code runs the torch-free `FakeBackend` (CI / plumbing +
attention-reconstruction quality) and, later, a real `HFBackend` (real answer
accuracy).

## Run it

```bash
# default matrix on the FakeBackend (no GPU / torch needed)
.venv/bin/python -m dexa.cli bench

# customize
.venv/bin/python -m dexa.cli bench --lengths 256,1024 --ratios 2,4,8,16 --n-per 2
.venv/bin/python -m dexa.cli bench --compactors random_subset,attention_matching
.venv/bin/python -m dexa.cli bench --no-plots          # tables only
.venv/bin/python -m dexa.cli bench --out-dir /tmp/run  # custom output dir

# real model (only if dexa.engine.hf + [torch] extras are installed)
.venv/bin/python -m dexa.cli bench --backend hf --model meta-llama/Llama-3.1-8B
```

Outputs land in `benchmarks/out/`:

- `results.json` — raw per-task rows + the cost model used.
- `frontier.png` — the money plot: recon error vs compression ratio, one line
  per compactor.
- `memory_saving.png` — KV memory saving per compactor × ratio.

## Tasks (`dexa.bench.tasks`)

RULER-style long-context probes; `length` is the count of *filler* tokens, with
needle/fact sentences spliced on top.

- `niah_single` — needle in a haystack: one planted
  "The magic number for `<key>` is `<value>`."; ask the value.
- `niah_multikey` — several needles with distinct keys; ask one (selectivity).
- `multihop` — variable tracking `X1=7; X2=X1; ...`; ask the final value.
- `synthetic_qa` — a few facts buried in filler + a question.

`make_tasks(backend, lengths=[...], n_per=...)` builds a flat list across
generators × lengths × seeds.

## Metrics (`dexa.bench.metrics`)

**Quality**

- `attention_recon_error` (model-free; works with `FakeBackend`) — compares the
  locally-normalized attention output of held-out reference queries over the
  full vs compact cache (the exact objective attention matching optimizes).
  Returns `cosine` (1.0 = perfect) and `rel_l2` (0.0 = perfect). This is the
  primary y-axis of the frontier.
- `answer_accuracy` (real-model signal) — greedy-generates an answer against the
  (possibly compact) context and scores exact-match + token-F1 vs the gold. On
  the toy `FakeBackend` this is not a language model, so accuracy stays ~0; it
  becomes meaningful with a real HF backend.

**System** (via `CostModel`)

- `compression_ratio` — `T / mean compact tokens per layer per kv-head`.
- `memory_saving` — `1 - kv_bytes(compact) / kv_bytes(full)`.
- `compaction_seconds` — measured wall time to compact.
- `decode_gpu_seconds_*` — modeled decode cost, proportional to attended cache
  length (each decoded token attends the whole cache).
- `recompute_avoided_seconds` — prefill you skip by reusing a stored compact
  cache instead of re-prefilling the context.

## Runner (`dexa.bench.runner`)

`run_matrix(backend, compactors, ratios, tasks, ref_strategy="repeat_prefill")`:
prefills each task once, derives reference queries once and splits them into a
**compaction set** (the compactor may see it) and a **held-out eval set** (used
only for scoring — no leakage), then sweeps each compactor × ratio cell. A
`FullKV` reference row (recon 0, compression 1) is always emitted per task so
tables and the frontier have an anchor. Results are plain dicts (no pandas) saved
to `results.json`.

## Compactors

The harness prefers the real registry
(`dexa.compaction.baselines`: `COMPACTORS`, `build(name, **kwargs)`). Until that
lands, `dexa.bench._compactors` provides a correct fallback implementing the
`Compactor` interface, resolved per-name: `full_kv`, `random_subset`,
`recent_window`, and `attention_matching` (keeps the keys carrying the most
attention mass over the reference queries — the leverage core of real attention
matching, which reliably beats random selection on reconstruction).

## Tests

```bash
.venv/bin/python -m pytest tests/test_bench.py -v
```
