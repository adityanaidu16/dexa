"""Dexa's actual wedge, benchmarked: resume a full session on a COLD instance vs
re-prefilling it from scratch.

Unlike prefix *sharing* (LMCache's game; `modal_bench_serve.py` showed Dexa is
mis-targeted there), this measures **session persistence / portability** — the novel
thing Dexa does: a specific long context, saved by one vLLM process, is resumed on a
*separate cold* vLLM process with ~0 re-prefill. The alternative on a cold box is a
full re-prefill (what a stateless engine pays after a preemption / replica move).

Three fresh containers (= three separate processes) share a persistent Volume store,
each timing time-to-first-token on the SAME long context:

  * populate — Dexa instance A: empty store, prefills + SAVES the context.  (setup)
  * cold     — VANILLA vLLM (no connector): full re-prefill.  → t_cold  (honest baseline)
  * resume   — fresh Dexa instance B: store has the context, LOADS it.     → t_resume

Headline: t_resume vs t_cold (both on cold instances). The win grows with context
length (prefill is ~O(n^2) compute; load is O(n) I/O) — see `docs/RESULTS.md`.

    modal run scripts/modal_bench_resume.py
    modal run scripts/modal_bench_resume.py --model Qwen/Qwen3-0.6B --ctx-len 32768
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_RESUME_GPU", "A10G")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "numpy")
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("src", "/root/dexa/src")
)

app = modal.App("dexa-bench-resume")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)
kv_store = modal.Volume.from_name("dexa-resume-store", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=3600,
              volumes={"/cache/hf": hf_cache, "/kv": kv_store})
def measure(model: str, ctx_len: int, label: str, use_connector: bool) -> dict:
    import glob
    import time

    from vllm import LLM, SamplingParams

    kv_store.reload()
    before = [os.path.basename(f) for f in glob.glob("/kv/*")]

    # Force SINGLE-STEP prefill (batch the whole prompt) so it lands in
    # scheduled_new_reqs and the connector's save fires — the connector does not yet
    # handle chunked prefill (long prompts spread over steps -> scheduled_cached_reqs),
    # which is a critical gap for real long sessions (see docs/CONNECTOR_COMPLETION.md).
    kwargs = dict(model=model, enforce_eager=True, gpu_memory_utilization=0.85,
                  max_model_len=ctx_len + 256, enable_prefix_caching=False,
                  max_num_batched_tokens=ctx_len + 256)
    if use_connector:
        from vllm.config import KVTransferConfig
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="DexaConnector",
            kv_connector_module_path="dexa.engine.vllm_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"dexa_store_root": "/kv"},
        )
    llm = LLM(**kwargs)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # deterministic token-id prompt of EXACT length (same ids across phases -> same
    # connector key). ids in a safe mid-vocab range, avoiding specials.
    ids = [5 + (i % 2000) for i in range(ctx_len)]
    warm = {"prompt_token_ids": [5, 6, 7, 8, 9, 10, 11, 12]}
    prompt = {"prompt_token_ids": ids}

    llm.generate([warm], sp)  # warm up CUDA kernels so they're not in the timing
    t0 = time.perf_counter()
    llm.generate([prompt], sp)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    kv_store.commit()
    after = [os.path.basename(f) for f in glob.glob("/kv/*")]
    print(f"[{label}] ctx={ctx_len} ttft={dt_ms:.0f}ms  store_before={len(before)} store_after={len(after)}",
          flush=True)
    return {"label": label, "ttft_ms": dt_ms, "before": before, "after": after}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-0.6B", ctx_len: int = 8192) -> None:
    print(f"resume benchmark on {GPU}: {model}, ctx_len={ctx_len}")
    pop = measure.remote(model, ctx_len, "populate(dexa save)", True)
    cold = measure.remote(model, ctx_len, "cold(vanilla re-prefill)", False)
    resume = measure.remote(model, ctx_len, "resume(dexa load)", True)

    saved = len(pop["after"]) > 0
    hit = len(resume["before"]) > 0
    tc, tr = cold["ttft_ms"], resume["ttft_ms"]
    speedup = tc / tr if tr > 0 else float("inf")
    print("\n" + "=" * 66)
    print(f"SESSION RESUME @ ctx={ctx_len} ({model}, {GPU})")
    print(f"  cold  (vanilla full re-prefill) : {tc:8.0f} ms")
    print(f"  resume(dexa load, cold instance): {tr:8.0f} ms")
    print(f"  -> resume speedup vs cold re-prefill: {speedup:.2f}x")
    print(f"  (A saved KV={saved}; resume instance saw stored KV on entry={hit})")
    print("=" * 66)
