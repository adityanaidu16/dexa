"""Stateful warm-session vs stateless re-prefill: does restoring KV beat recomputing it?

The stateful-inference thesis rests on one measurable physical fact. When a long-lived agent
session goes idle (past a stateless provider's prompt-cache TTL) and then resumes, is it
cheaper and faster to RESTORE its saved KV cache from CPU/NVMe than to RE-PREFILL the whole
context from scratch? Re-prefill is compute-bound and grows with context; restore is
bandwidth-bound (move bytes, don't recompute them). If restore << prefill for large contexts,
a session-stateful architecture has a real, physical reason to exist. If not, it doesn't.

For a range of context sizes we measure, on one A100-80GB (Qwen2.5-7B, bf16):
  prefill_ms    forward pass over C tokens to build the KV   -> what STATELESS pays on resume
  offload_ms    move that KV  GPU -> pinned CPU
  restore_cpu   move it       pinned CPU -> GPU              -> what STATEFUL pays (warm tier)
  restore_disk  NVMe file -> GPU (load + H2D)               -> stateful, offloaded/cold tier
  decode_ms     one decode step given the KV                -> same for both; the marginal turn
Correctness: after a CPU round-trip we decode one token and check the KV still advances.
Then we project a T-turn, idle-gapped session from the measured primitives.

    modal run evals/modal_stateful_session.py
"""

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CTX_SIZES = [4096, 16384, 32768, 65536, 131072]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "transformers==4.49.0", "accelerate")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)
app = modal.App("dexa-stateful-session")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=2400, volumes={"/cache/hf": hf_cache})
def run() -> None:
    import gc
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    vocab = model.config.vocab_size

    def sync():
        torch.cuda.synchronize()

    @torch.inference_mode()
    def prefill(C):
        ids = torch.randint(100, vocab - 100, (1, C), device=dev)
        sync(); t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True, num_logits_to_keep=1)
        sync(); dt = time.perf_counter() - t0
        return out.past_key_values, dt, ids

    @torch.inference_mode()
    def decode_step(past, last_id):
        sync(); t0 = time.perf_counter()
        out = model(input_ids=last_id, past_key_values=past, use_cache=True, num_logits_to_keep=1)
        sync(); return out.past_key_values, time.perf_counter() - t0, out.logits

    def legacy(past):
        return past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past

    def kv_bytes(tuples):
        return sum(t.numel() * t.element_size() for layer in tuples for t in layer)

    # warmup (compile kernels, allocate caches) so the first real timing isn't skewed
    p, _, ids = prefill(2048)
    decode_step(p, ids[:, -1:])
    del p; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print(f"STATEFUL WARM-SESSION vs RE-PREFILL  ·  {MODEL}  ·  A100-80GB  ·  bf16")
    print("=" * 96)
    hdr = f"{'context':>9} {'KV size':>9} {'prefill':>10} {'offload':>9} {'restore(CPU)':>13} {'restore(NVMe)':>14} {'decode/step':>12}"
    print(hdr); print("-" * 96)

    rows = []
    for C in CTX_SIZES:
        try:
            past, prefill_ms, ids = prefill(C)
            prefill_ms *= 1000
            tuples = legacy(past)
            nbytes = kv_bytes(tuples)

            # offload GPU -> pinned CPU
            sync(); t0 = time.perf_counter()
            cpu_tuples = tuple(tuple(t.to("cpu", non_blocking=True) for t in layer) for layer in tuples)
            sync(); offload_ms = (time.perf_counter() - t0) * 1000

            # free the GPU copy so restore is a real re-materialization
            del past, tuples
            gc.collect(); torch.cuda.empty_cache()

            # restore pinned CPU -> GPU
            cpu_tuples = tuple(tuple(t.pin_memory() for t in layer) for layer in cpu_tuples)
            sync(); t0 = time.perf_counter()
            gpu_tuples = tuple(tuple(t.to(dev, non_blocking=True) for t in layer) for layer in cpu_tuples)
            sync(); restore_cpu_ms = (time.perf_counter() - t0) * 1000

            # correctness: KV restored -> one decode step advances the cache to len C+1
            restored = DynamicCache.from_legacy_cache(gpu_tuples)
            before = restored.get_seq_length()
            _, dstep, _ = decode_step(restored, ids[:, -1:])
            after = restored.get_seq_length()
            assert after == before + 1, f"KV did not advance after restore ({before}->{after})"
            decode_ms = dstep * 1000

            # NVMe tier: serialize to local disk, then load + H2D
            path = f"/tmp/kv_{C}.pt"
            torch.save(cpu_tuples, path)
            del gpu_tuples, restored
            gc.collect(); torch.cuda.empty_cache()
            sync(); t0 = time.perf_counter()
            loaded = torch.load(path, map_location="cpu")
            disk_gpu = tuple(tuple(t.to(dev) for t in layer) for layer in loaded)
            sync(); restore_disk_ms = (time.perf_counter() - t0) * 1000
            os.remove(path)

            rows.append((C, nbytes, prefill_ms, offload_ms, restore_cpu_ms, restore_disk_ms, decode_ms))
            print(f"{C:>9} {nbytes/1e9:>7.2f}GB {prefill_ms:>9.0f}ms {offload_ms:>8.0f}ms "
                  f"{restore_cpu_ms:>11.0f}ms {restore_disk_ms:>12.0f}ms {decode_ms:>10.1f}ms")

            del cpu_tuples, disk_gpu, loaded
            gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{C:>9}   OOM — skipped")
            gc.collect(); torch.cuda.empty_cache()

    print("=" * 96)
    print("\nRESUME-AFTER-IDLE  (the decisive comparison: stateless re-prefills, stateful restores)")
    print(f"  {'context':>9} {'prefill(stateless)':>19} {'restore CPU(stateful)':>22} {'speedup':>9} {'NVMe speedup':>13}")
    for C, _b, pf, _o, rc, rd, _d in rows:
        print(f"  {C:>9} {pf:>17.0f}ms {rc:>20.0f}ms {pf/rc:>7.1f}x {pf/rd:>11.1f}x")

    # project a long-lived session: C context, T turns, each turn preceded by an idle gap that
    # would evict a stateless cache -> stateless re-prefills every turn; stateful restores.
    print("\nPROJECTED SESSION  (T turns, each resumed after an idle gap that evicts a stateless cache)")
    print(f"  {'context':>9} {'turns':>6} {'stateless prefill-time':>23} {'stateful restore-time':>22} {'multiple':>9}")
    for C, _b, pf, _o, rc, _rd, _d in rows:
        for T in (10, 50):
            stateless = T * pf
            stateful = pf + T * rc          # prefill once, then restore each resume
            print(f"  {C:>9} {T:>6} {stateless:>21.0f}ms {stateful:>20.0f}ms {stateless/stateful:>7.1f}x")
    print("=" * 96)
    print("Note: restore(CPU) assumes the warm/pinned tier; restore(NVMe) includes torch.load")
    print("deserialization (an upper bound — a real system uses zero-copy/direct IO). Within a")
    print("stateless cache's TTL there is NO idle gap, so stateless ~= stateful; the win is the")
    print("idle-gap / eviction tail, which is exactly where long-lived agents live.")
    hf_cache.commit()


@app.local_entrypoint()
def main() -> None:
    run.remote()
