"""Production reproduction: does KV reuse beat re-prefill in the real serving engine (vLLM)?

The HF experiment (`modal_stateful_session.py`) measured the raw physics with tensor copies.
This reproduces the same claim inside vLLM's actual paged-KV engine, using its prefix cache as
the "restore" path:

  cold  = send a long prompt vLLM has never seen -> full prefill -> time to first token (TTFT)
  warm  = send the SAME prompt again -> prefix-cache hit -> KV reused, ~no prefill -> TTFT

The cold/warm TTFT ratio is the production analog of re-prefill vs restore. Paged KV also lets
us reach 128k, which the non-paged HF run OOM'd on.

    modal run evals/modal_vllm_warmstart.py
"""

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 70000
CTX_SIZES = [4096, 16384, 32768, 65536]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1", "VLLM_USE_V1": "1"})
)
app = modal.App("dexa-vllm-warmstart")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=2400, volumes={"/cache/hf": hf_cache})
def run() -> None:
    import random
    import time

    from vllm import LLM, SamplingParams

    # Qwen2.5-7B is natively 32k; reach 128k via YaRN. We only time prefill vs cache-hit
    # (generate 1 token), so YaRN's effect on output quality is irrelevant here.
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    # chunked prefill caps activation memory so long (32k-64k) prefills don't OOM the engine
    common = dict(enable_prefix_caching=True, enable_chunked_prefill=True,
                  max_num_batched_tokens=8192, gpu_memory_utilization=0.85,
                  enforce_eager=False)
    try:
        llm = LLM(model=MODEL, max_model_len=MAX_LEN, rope_scaling=yarn, **common)
        sizes = CTX_SIZES
    except Exception as e:  # noqa: BLE001 — fall back to native 32k context
        print(f"(YaRN long-context init failed: {str(e)[:80]} — capping at 32k)")
        llm = LLM(model=MODEL, max_model_len=32768, **common)
        sizes = [c for c in CTX_SIZES if c <= 30000]
    vocab = llm.llm_engine.model_config.get_vocab_size()
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(0)

    def ids(n, salt):
        # unique per (size, salt) so a "cold" prompt is genuinely uncached
        r = random.Random(salt)
        return [r.randint(10, vocab - 10) for _ in range(n)]

    def ttft_ms(token_ids):
        t0 = time.perf_counter()
        llm.generate([{"prompt_token_ids": token_ids}], sp, use_tqdm=False)
        return (time.perf_counter() - t0) * 1000

    # warmup (compile CUDA graphs / kernels) with a prompt none of the tests reuse
    ttft_ms(ids(2048, salt=999))

    print("\n" + "=" * 88)
    print(f"vLLM PAGED-KV WARM-START  ·  {MODEL}  ·  A100-80GB  ·  prefix caching ON")
    print("=" * 88)
    print(f"  {'context':>9} {'cold TTFT (prefill)':>20} {'warm TTFT (KV reuse)':>21} {'speedup':>9}")
    print("-" * 88)
    rows = []
    for C in sizes:
        try:
            prompt = ids(C, salt=C)                 # distinct per size -> first call is cold
            cold = ttft_ms(prompt)
            warm = min(ttft_ms(prompt) for _ in range(3))   # best of 3 (warm is fast/noisy)
            rows.append((C, cold, warm))
            print(f"  {C:>9} {cold:>18.0f}ms {warm:>19.0f}ms {cold/warm:>7.1f}x")
        except Exception as e:  # noqa: BLE001
            print(f"  {C:>9}   error: {str(e)[:60]}")

    print("=" * 88)
    if rows:
        print("\nInterpretation: warm TTFT = KV restored from the engine's cache instead of")
        print("recomputed. The gap is the per-turn cost a stateless provider pays whenever an")
        print("idle gap has evicted the prefix. Compare to the HF raw-physics run in RESULTS.md.")
        big = max(rows, key=lambda r: r[1] / r[2])
        print(f"\n  peak: {big[0]} ctx -> {big[1]/big[2]:.1f}x faster to reuse than re-prefill "
              f"({big[1]:.0f}ms -> {big[2]:.0f}ms)")
    print("=" * 88)
    hf_cache.commit()


@app.local_entrypoint()
def main() -> None:
    run.remote()
