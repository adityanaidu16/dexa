# Running Dexa on a GPU cluster

This is the runbook for reproducing Dexa's benchmarks on a CUDA box — no code
edits required. Everything is config-driven (`dexa run --config <file>`); the
scripts in `scripts/` and `scripts/slurm/` wrap the same entrypoint for bare
nodes and SLURM. For what the numbers mean and the honest caveats, read
[`docs/RESULTS.md`](RESULTS.md).

---

## 1. Prerequisites

- A Linux node with an NVIDIA GPU and a recent driver.
- Python 3.10+.
- A **CUDA build of PyTorch** matching your driver (the default PyPI wheel is
  CPU-only). Install it from the CUDA index first, then the project:

```bash
# pick the CUDA tag that matches your driver: cu121, cu124, ...
pip install torch --index-url https://download.pytorch.org/whl/cu121

# then the project + benchmark/reporting extras (editable)
pip install -e '.[torch,bench]'

# optional: vLLM backend for multi-GPU throughput (needs a CUDA torch)
pip install -e '.[gpu]'
```

Or use the bootstrap script, which does the above into a repo-local `.venv` and
is safe to re-run:

```bash
./scripts/setup_env.sh                 # cu121 + [torch,bench]
CUDA=cu124 ./scripts/setup_env.sh      # different CUDA wheel
./scripts/setup_env.sh --vllm          # also install the [gpu] extra
```

---

## 2. Hugging Face auth & model download

The example configs reference Meta's Llama models, which are **gated** on the
Hub. Either request access and authenticate, or point at an ungated mirror.

```bash
# gated meta-llama models: authenticate with a token that has access
export HF_TOKEN=hf_xxx

# if model downloads stall (HF's xet transfer protocol hangs in some envs):
export HF_HUB_DISABLE_XET=1
```

Ungated mirrors (drop-in, no token needed) if you don't have Llama access:

- `unsloth/Llama-3.2-1B-Instruct`
- `unsloth/Llama-3.2-3B-Instruct`
- `unsloth/Meta-Llama-3.1-8B-Instruct`

Just set `model:` in the config to the mirror id. `dexa run` only loads the
model when it actually runs — parsing/validating a config never fetches weights.

---

## 3. Running a benchmark

```bash
# direct
dexa run --config configs/llama32-1b.yaml

# wrapper (sets HF_HUB_DISABLE_XET=1, passes extra args through)
./scripts/run_bench.sh configs/llama32-1b.yaml
```

Outputs land in the config's `out_dir` (e.g. `benchmarks/out/llama32-1b/`):

| file | what it is |
|---|---|
| `niah.json` | raw needle-recall results (per method/ratio/seed) |
| `agentic.json` | raw long-horizon agentic trajectory results |
| `results.json` | the combined run record (config + both suites) |
| `REPORT.md` | human-readable summary tables |
| `niah_frontier.png` | recall-vs-compression-ratio plot per method |

**Reading them.** `REPORT.md` is the place to start: the needle-recall table
shows recall per compactor and ratio (1.0 = full-KV, 0.0 = no context) plus the
paired `AM−HH` delta and the fraction of seeds where AM wins; the agentic table
shows late-recall vs peak memory vs compaction count. For exact metric
definitions (including the `recall_frac` rescaling and the `>1.0` denoising
artifact) and the statistical reading, see [`docs/RESULTS.md`](RESULTS.md).

---

## 4. Scaling up

The provided scale-up configs are copies of the 1B config with the model swapped
and the cost model adjusted:

- `configs/llama32-3b.yaml` — `meta-llama/Llama-3.2-3B-Instruct`
- `configs/llama31-8b.yaml` — `meta-llama/Llama-3.1-8B-Instruct`, with longer
  contexts (`lengths: [4000, 16000, 32000]`) to probe the 32k regime.

To go bigger or longer, edit `model:` and `niah.lengths:` in any config.

**Throughput via vLLM + tensor parallelism.** The `HFBackend` (`backend: hf`) is
the faithful, deterministic eval path and runs single-GPU. For 8B-class models
and many seeds, switch to the cluster backend by setting `backend: vllm` and
launching with tensor parallelism across GPUs (`--tensor-parallel-size N`, more
GPUs per node in your SLURM directives). Note (per `docs/RESULTS.md` §5) that the
vLLM compact-decode path needs a site-specific beta-aware attention backend wired
up before compact `generate`/`score` run; the `HFBackend` remains the reference.

---

## 5. SLURM

Batch scripts live in `scripts/slurm/`. They request 1 GPU, activate the repo
`.venv` (created by `setup_env.sh`), and `srun` the CLI. Module/conda lines are
commented placeholders — edit them for your site.

```bash
# default config (configs/llama32-1b.yaml)
sbatch scripts/slurm/bench.sbatch

# override the config via --export or an env var
sbatch --export=ALL,CONFIG=configs/llama32-3b.yaml scripts/slurm/bench.sbatch
CONFIG=configs/llama31-8b.yaml sbatch scripts/slurm/bench.sbatch
```

Set `HF_TOKEN` in your submission environment (e.g.
`--export=ALL,HF_TOKEN=...,CONFIG=...`) for gated models.

---

## 6. STILL training

STILL learns the per-layer perceivers once against a frozen base model so that
later compaction is a single forward pass. Hyperparameters are captured in
`configs/still-train.yaml`; every key maps to a trainer CLI flag (an explicit
flag overrides the config, which overrides the built-in smoke defaults).

```bash
# real run from config
python -m dexa.compaction.still.train --config configs/still-train.yaml

# under SLURM
sbatch --export=ALL,CONFIG=configs/still-train.yaml scripts/slurm/train_still.sbatch

# tiny CPU smoke (no config): device defaults to cuda but falls back to cpu
python -m dexa.compaction.still.train --steps 2 --n-samples 1
```

`--device` defaults to `cuda` and falls back to CPU automatically when no GPU is
present; `--dtype auto` picks float32 on CPU and bfloat16 on GPU. The in-repo
trainer distills against synthetic random samples — the loop, optimizer and loss
are identical to a full run, but a production run should swap in a real document
sampler (see `docs/RESULTS.md` §5 on STILL's training status).

---

## 7. Interpreting results (the honest headline)

From `docs/RESULTS.md`, stated plainly and non-promotionally:

- **AM's robust, statistically significant win is at extreme compression
  (128x):** Attention Matching keeps the needle where heavy-hitter (H2O)
  collapses and SnapKV degrades — a large effect, unanimous across seeds, and
  significant.
- **At mid ratios (4–32x) AM and H2O are effectively tied.** The metric
  saturates (every reasonable method lands ~0.95–1.0), and some independent runs
  show H2O ahead at 16–64x. AM also costs ~5x more compaction compute. A claim
  that AM "beats H2O at 4–32x" is **not** supported by the data.
- **In the agentic setting** (moderate per-compaction ratios) H2O slightly
  outperforms AM on late recall and is cheaper, consistent with the above — AM's
  edge is at extreme compression, which that trajectory does not reach.

So: AM ≥ H2O, AM degrades far more gracefully at ≥64–128x, and at mid ratios pick
the cheaper method. Whether the 128x advantage holds on 8B-class models with long
contexts is exactly what the scale-up configs here are for — and is still
unverified (`docs/RESULTS.md` §5).
