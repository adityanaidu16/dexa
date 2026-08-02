"""The decisive gate: does SGLang (RadixAttention) also blow up on long-context
branched decode, or does its prefix-sharing tree already solve it?

vLLM showed an 8-wide decode over a long shared prefix costs 174x a single decode at
128k (ideal 8x) — the shared prefix KV is re-read per branch. SGLang's RadixAttention
stores the shared prefix once in a radix tree, which is the closest existing thing to a
"tree-native" inference stack. If SGLang ALSO blows up superlinearly, the architectural
white space (an efficient shared-prefix branched decode) is real. If it's ~linear, the
space is already occupied and there's no reason to build a new stack.

Same measurement as modal_decode_pathology.py: per-step decode cost across context length
L and branch count n, via a batch of n identical L-token prompts (RadixAttention shares
the prefix). decode/step = (t_full - t_prefill)/(gen-1); ratio = per_step(n)/per_step(1).

cuda graph is off (fast startup, no capture OOM at 128k); it only removes per-step launch
overhead, not the attention scaling the pathology is about, so the ratio is unaffected.

    modal run evals/modal_sglang_pathology.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

# SGLang pulls a matched torch + flashinfer; build on a CUDA devel base for nvcc.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("sglang[all]", "transformers")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-sglang-pathology")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

CONTEXT_LENS = [4096, 32768, 130944]
N_SWEEP = [1, 2, 4, 8]
PARA = (
    "The quarterly review covered revenue, churn, and the roadmap for the data platform. "
    "Engineering flagged latency regressions in the ingestion pipeline and proposed a "
    "migration to a streaming architecture with backpressure and exactly-once delivery. "
)


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, gen: int) -> None:
    from time import perf_counter

    import sglang as sgl
    from transformers import AutoTokenizer

    print(f"[info] sglang {getattr(sgl, '__version__', '?')}", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    base_ids = [int(x) for x in tok(PARA, add_special_tokens=False)["input_ids"]]

    def ids_for(L):
        reps = L // len(base_ids) + 1
        return (base_ids * reps)[:L]

    engine = sgl.Engine(model_path=model, mem_fraction_static=0.85,
                        disable_cuda_graph=True, context_length=max(CONTEXT_LENS) + gen + 8)

    G1, G2 = 8, gen + 8  # difference isolates (G2-G1) decode steps

    def branches(L, n):
        # n distinct sequences sharing the same length-(L-1) prefix: a unique final
        # token per branch prevents request dedup, while RadixAttention shares the
        # common prefix. This is the real branch scenario.
        pref = ids_for(L - 1)
        return [pref + [100 + i] for i in range(n)]

    def gen_time(batch, mnt):
        sp = [{"temperature": 0.8, "top_p": 0.95, "max_new_tokens": mnt} for _ in batch]
        t = perf_counter()
        engine.generate(input_ids=batch, sampling_params=sp)
        return perf_counter() - t

    def decode_per_step(L, n):
        batch = branches(L, n)
        gen_time(batch, 4)            # warm the shared prefix + JIT
        t1 = gen_time(batch, G1)
        t2 = gen_time(batch, G2)
        return max((t2 - t1) / (G2 - G1), 1e-6)

    branches_warm = branches(512, 2)
    gen_time(branches_warm, 8)        # global warmup

    results = {}
    for L in CONTEXT_LENS:
        per_step = {}
        for n in N_SWEEP:
            per_step[n] = decode_per_step(L, n)
            print(f"[L={L:>7} n={n}] decode/step={per_step[n]*1000:8.2f}ms", flush=True)
        results[L] = per_step

    print("\n" + "=" * 80)
    print(f"SGLANG DECODE-PATHOLOGY — {model} ({GPU}), gen={gen}, RadixAttention on")
    print("=" * 80)
    print(f"  {'context L':>10} {'n':>3} {'decode/step ms':>15} {'ratio vs n=1':>14} {'ideal':>7}")
    for L in CONTEXT_LENS:
        base = results[L][1]
        for n in N_SWEEP:
            r = results[L][n] / base
            flag = "  <-- BLOWUP" if r > 1.5 * n else ""
            print(f"  {L:>10} {n:>3} {results[L][n]*1000:>15.2f} {r:>13.2f}x {n:>6}x{flag}")
        print()
    r_short = results[CONTEXT_LENS[0]][8] / results[CONTEXT_LENS[0]][1]
    r_long = results[CONTEXT_LENS[-1]][8] / results[CONTEXT_LENS[-1]][1]
    print("=" * 80)
    print(f"  HEADLINE: SGLang 8-wide decode is {r_short:.1f}x a single decode at "
          f"{CONTEXT_LENS[0]//1024}k and {r_long:.1f}x at {CONTEXT_LENS[-1]//1024}k "
          f"(ideal 8x). vLLM was 4.6x -> 174x. "
          f"{'SGLang ALSO blows up: white space real' if r_long > 12 else 'SGLang handles it: space occupied'}.")
    print("=" * 80)
    engine.shutdown()
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", gen: int = 64) -> None:
    print(f"SGLang decode-pathology sweep on {GPU}: {model}")
    run.remote(model, gen)
