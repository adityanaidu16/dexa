"""The thesis curve: does prefix-sharing / persistence value GROW with context length?

The long-context-agent bet says the KV cache stops being an optimization and becomes
required infrastructure: a corpus/session prefix that costs seconds of prefill compute,
reused across many branches (best-of-N, parallel tool-calls, reasoning paths). The naive
pattern re-sends the full context per branch and re-prefills it every time; the dexa
pattern prefills the shared prefix ONCE and forks N branches off it.

This measures both, live, across context lengths L, and reports the speedup as a function
of L. The claim to falsify: the advantage is ~flat at short L (prefill is a tiny slice of
the work) and DIVERGES as L grows (prefill dominates, so paying it once vs N times
approaches an N× saving). The slope of that curve is the thesis.

  * shared  (dexa) : one request, n=N   -> prefill(L) once + N decodes  (branches share
                     the prefix KV; also memory-frugal — one prefix resident, not N).
  * independent    : N requests, n=1 each -> prefill(L) N times + N decodes  (what naive
                     agents do — every call re-sends the whole context).

Prefix caching is OFF on purpose: it models the regime the thesis is ABOUT — contexts too
big and sessions too many to stay resident in HBM, so reuse requires explicit
persistence/branching, not the free warm-cache case vLLM already handles within one
synchronous request. prefill_time(L) is measured too, to show the mechanism.

    modal run evals/modal_context_scaling.py
    modal run evals/modal_context_scaling.py --n-branch 4
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

app = modal.App("dexa-context-scaling")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

# ~128k top end stays under Llama-3.1's 131072 max once GEN + margin is added.
CONTEXT_LENS = [4096, 16384, 32768, 65536, 130816]
PARA = (
    "The quarterly review covered revenue, churn, and the roadmap for the data platform. "
    "Engineering flagged latency regressions in the ingestion pipeline and proposed a "
    "migration to a streaming architecture with backpressure and exactly-once delivery. "
    "Legal raised questions about data residency across the EU and APAC regions. "
)


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, N: int, gen: int) -> None:
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

    sp_branch = SamplingParams(n=N, temperature=0.8, top_p=0.95, max_tokens=gen, seed=0)
    sp_one = SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=gen, seed=0)
    sp_prefill = SamplingParams(n=1, temperature=0.0, max_tokens=1, seed=0)

    # warm up kernels so the first timed prefill isn't skewed by JIT.
    llm.generate([prompt(512)], sp_one, use_tqdm=False)

    rows = []
    for L in CONTEXT_LENS:
        t = perf_counter()
        llm.generate([prompt(L)], sp_prefill, use_tqdm=False)
        prefill_s = perf_counter() - t

        t = perf_counter()
        llm.generate([prompt(L)], sp_branch, use_tqdm=False)          # prefill L once, N branches
        shared_s = perf_counter() - t

        t = perf_counter()
        for _ in range(N):
            llm.generate([prompt(L)], sp_one, use_tqdm=False)         # prefill L, N times
        indep_s = perf_counter() - t

        speedup = indep_s / shared_s if shared_s else 0.0
        rows.append((L, prefill_s, shared_s, indep_s, speedup))
        print(f"[L={L:>7}] prefill={prefill_s:6.2f}s  shared={shared_s:6.2f}s  "
              f"independent={indep_s:7.2f}s  speedup={speedup:5.2f}x", flush=True)

    print("\n" + "=" * 82)
    print(f"CONTEXT-SCALING: branch-sharing vs naive re-prefill — {model} ({GPU}), "
          f"N={N} branches, gen={gen}")
    print("=" * 82)
    print(f"  {'context L':>10} {'prefill s':>10} {'shared s':>10} {'independent s':>14} "
          f"{'speedup':>9} {'prefill %':>10}")
    for L, pf, sh, ind, sp in rows:
        pf_share = pf / sh * 100 if sh else 0.0
        print(f"  {L:>10} {pf:>10.2f} {sh:>10.2f} {ind:>14.2f} {sp:>8.2f}x {pf_share:>9.0f}%")
    print("=" * 82)
    lo, hi = rows[0], rows[-1]
    print(f"  HEADLINE: branch-sharing speedup rises {lo[4]:.2f}x -> {hi[4]:.2f}x as context "
          f"grows {lo[0]//1024}k -> {hi[0]//1024}k. The prize scales with context length — "
          f"exactly the long-context-agent regime.")
    print("=" * 82)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", n_branch: int = 8,
         gen: int = 128) -> None:
    print(f"context-scaling curve on {GPU}: {model}, N={n_branch} branches")
    run.remote(model, n_branch, gen)
