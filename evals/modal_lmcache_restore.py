"""Offload -> restore through a real engine: vLLM + LMCache, KV evicted to CPU then restored.

The in-GPU prefix-cache reproduction (`modal_vllm_warmstart.py`) proved reuse-beats-recompute
while KV stays on the GPU. The stateful thesis needs the harder case: KV **offloaded off the
GPU** (an idle session evicted from HBM) and then **restored** on resume. LMCache is the
production KV-offload store; this wires it into vLLM with vLLM's own prefix cache OFF, so the
ONLY way request #2 can be fast is if LMCache restored the KV from CPU.

  cold     = request #1, long prefix, never seen -> full prefill (LMCache stores KV to CPU)
  restore  = request #2, same prefix -> vLLM would re-prefill, but LMCache loads KV from CPU

If restore << cold with APC off, the offload->restore path works in a real serving stack.

    modal run evals/modal_lmcache_restore.py
"""

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CTX_SIZES = [4096, 16384]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "lmcache")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1", "VLLM_USE_V1": "1",
          # LMCache: CPU RAM backend, generous budget, small chunks for fine-grained reuse
          "LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "20",
          "LMCACHE_CHUNK_SIZE": "256"})
)
app = modal.App("dexa-lmcache-restore")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=2400, volumes={"/cache/hf": hf_cache})
def run() -> None:
    import random
    import time

    # NB: do NOT import lmcache here — it inits CUDA in the parent and poisons vLLM's engine
    # fork. vLLM loads the LMCache connector inside the worker process itself.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    # APC OFF -> vLLM will not reuse in-GPU blocks; LMCache (CPU) is the only reuse path.
    kv = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(model=MODEL, enable_prefix_caching=False, kv_transfer_config=kv,
              max_model_len=20000, gpu_memory_utilization=0.80,
              enable_chunked_prefill=True, max_num_batched_tokens=8192, enforce_eager=False)
    vocab = llm.llm_engine.model_config.get_vocab_size()
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def ids(n, salt):
        r = random.Random(salt)
        return [r.randint(10, vocab - 10) for _ in range(n)]

    def ttft_ms(token_ids):
        t0 = time.perf_counter()
        llm.generate([{"prompt_token_ids": token_ids}], sp, use_tqdm=False)
        return (time.perf_counter() - t0) * 1000

    ttft_ms(ids(2048, 999))  # warmup

    print("\n" + "=" * 88)
    print(f"vLLM + LMCache OFFLOAD->RESTORE  ·  {MODEL}  ·  APC OFF (LMCache is the only reuse)")
    print("=" * 88)
    print(f"  {'context':>9} {'cold (prefill)':>16} {'restore (LMCache CPU)':>22} {'speedup':>9}")
    print("-" * 88)
    for C in CTX_SIZES:
        try:
            p = ids(C, salt=C)
            cold = ttft_ms(p)
            time.sleep(1.0)                              # a small "idle gap"
            restore = min(ttft_ms(p) for _ in range(3))
            flag = "" if restore < cold * 0.7 else "  <-- no speedup (LMCache not restoring?)"
            print(f"  {C:>9} {cold:>14.0f}ms {restore:>20.0f}ms {cold/restore:>7.1f}x{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"  {C:>9}   error: {str(e)[:70]}")
    print("=" * 88)
    print("APC is OFF, so any speedup is KV restored from LMCache's CPU store, not an in-GPU")
    print("prefix hit — the offload->restore path the stateful-session thesis needs.")
    hf_cache.commit()


@app.local_entrypoint()
def main() -> None:
    run.remote()
