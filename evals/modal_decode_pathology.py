"""Confirm (or refute) the vLLM long-context branched-decode pathology.

The context-scaling run showed shared n=N branching stops helping at 128k because the
concurrent branched DECODE blows up. This isolates that: for each context length L and
branch count n, measure per-step decode cost and compare to a single-branch decode.

  decode_per_step(L, n) = ( t_full(L, n, gen) - t_prefill(L) ) / gen
  batching_ratio(L, n)  = decode_per_step(L, n) / decode_per_step(L, 1)

If parallel sampling is healthy, an n-wide decode over a SHARED prefix should cost
roughly the single-branch cost plus a little (the prefix KV is read once, reused across
branches) — ratio grows slowly, well under n. If ratio >> n and worsens with L, the
prefix KV is being re-read per branch: the pathology, and the product opening (a
shared-prefix / tree-attention decode that reads the long KV once).

enforce_eager is on (as in the first run); its fixed per-step overhead is additive and
n-independent, so it *compresses* the ratio toward 1 — i.e. it UNDER-states the
pathology. A blowup seen here is therefore conservative.

    modal run evals/modal_decode_pathology.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "transformers", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-decode-pathology")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

CONTEXT_LENS = [4096, 32768, 130944]   # ~4k, 32k, ~128k (stays under 131072 with gen)
N_SWEEP = [1, 2, 4, 8]
PARA = (
    "The quarterly review covered revenue, churn, and the roadmap for the data platform. "
    "Engineering flagged latency regressions in the ingestion pipeline and proposed a "
    "migration to a streaming architecture with backpressure and exactly-once delivery. "
)


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, gen: int) -> None:
    from time import perf_counter

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    base_ids = [int(x) for x in tok(PARA, add_special_tokens=False)["input_ids"]]

    def prompt(L: int) -> dict:
        reps = L // len(base_ids) + 1
        return {"prompt_token_ids": (base_ids * reps)[:L]}

    max_len = max(CONTEXT_LENS) + gen + 8
    llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=0.9,
              enforce_eager=True, enable_prefix_caching=False)

    sp_prefill = SamplingParams(n=1, temperature=0.0, max_tokens=1, seed=0)

    def sp(n):
        return SamplingParams(n=n, temperature=0.8, top_p=0.95, max_tokens=gen, seed=0)

    def timed(p, params):
        t = perf_counter()
        llm.generate([p], params, use_tqdm=False)
        return perf_counter() - t

    timed(prompt(512), sp(2))  # warmup kernels

    results = {}
    for L in CONTEXT_LENS:
        p = prompt(L)
        t_prefill = min(timed(p, sp_prefill) for _ in range(2))  # best-of-2, drop noise
        per_step = {}
        for n in N_SWEEP:
            t_full = timed(p, sp(n))
            per_step[n] = max((t_full - t_prefill) / gen, 1e-6)
            print(f"[L={L:>7} n={n}] full={t_full:7.2f}s prefill={t_prefill:6.2f}s "
                  f"decode/step={per_step[n]*1000:8.2f}ms", flush=True)
        results[L] = (t_prefill, per_step)

    print("\n" + "=" * 80)
    print(f"DECODE-PATHOLOGY SWEEP — {model} ({GPU}), gen={gen}, prefix-caching off, eager")
    print("=" * 80)
    print(f"  {'context L':>10} {'n':>3} {'decode/step ms':>15} {'ratio vs n=1':>14} {'ideal (=n)':>11}")
    for L in CONTEXT_LENS:
        _pf, per_step = results[L]
        base = per_step[1]
        for n in N_SWEEP:
            ratio = per_step[n] / base
            flag = "  <-- BLOWUP" if ratio > 1.5 * n else ""
            print(f"  {L:>10} {n:>3} {per_step[n]*1000:>15.2f} {ratio:>13.2f}x {n:>10}x{flag}")
        print()
    print("=" * 80)
    r_short = results[CONTEXT_LENS[0]][1][8] / results[CONTEXT_LENS[0]][1][1]
    r_long = results[CONTEXT_LENS[-1]][1][8] / results[CONTEXT_LENS[-1]][1][1]
    print(f"  HEADLINE: 8-wide decode costs {r_short:.1f}x a single decode at "
          f"{CONTEXT_LENS[0]//1024}k but {r_long:.1f}x at {CONTEXT_LENS[-1]//1024}k "
          f"(ideal is 8x). {'Pathology CONFIRMED' if r_long > 12 else 'No blowup'} — the "
          f"shared prefix KV is {'re-read per branch' if r_long > 12 else 'reused well'}.")
    print("=" * 80)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", gen: int = 64) -> None:
    print(f"decode-pathology sweep on {GPU}: {model}, gen={gen}")
    run.remote(model, gen)
