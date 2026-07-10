"""Validate contention-aware loading: does loading KV beat re-prefill under CONCURRENT
load, and does the adaptive policy pick the winner?

The thesis: under GPU contention a new request's re-prefill queues behind other work
and competes for compute, while a KV load uses (mostly) idle I/O — so loading should
win at shorter contexts when the GPU is busy, which single-request benchmarks miss.

Method: for each server config, POPULATE N distinct long sessions (sequential, so the
connector saves them), then REPLAY the same N contexts at high concurrency and measure
per-request TTFT (streaming). Compare P50/P99 TTFT and throughput across:

  * vanilla       — no connector: every request re-prefills (baseline under load).
  * dexa_always   — connector, policy=always: always load.
  * dexa_adaptive — connector, policy=adaptive, contention-aware: load only when the
                    busy GPU makes re-prefill the slower option.

Honest caveat baked in: the connector load is currently SYNCHRONOUS (start_load_kv
blocks the worker step), so loads may serialize like prefills — if dexa doesn't beat
vanilla here, the connector needs async loading to win under contention. Either result
is a real finding.

    modal run scripts/modal_bench_contention.py
    DEXA_CONT_GPU=A100-80GB modal run scripts/modal_bench_contention.py --model unsloth/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_CONT_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "numpy", "aiohttp")
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("src", "/root/dexa/src")
)

app = modal.App("dexa-bench-contention")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/cache/hf": hf_cache})
def bench(model: str, ctx_len: int, n_sessions: int) -> None:
    import asyncio
    import json
    import subprocess
    import time
    import urllib.request

    import aiohttp

    port = 8000
    url = f"http://localhost:{port}"

    def wait_ready(timeout=400):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                urllib.request.urlopen(f"{url}/health", timeout=2)
                return True
            except Exception:
                time.sleep(2)
        return False

    # N distinct long contexts as explicit token ids (exact-repeat -> whole-prompt hit).
    contexts = [[5 + ((i * 131 + j) % 2000) for j in range(ctx_len)] for i in range(n_sessions)]

    async def one(session, ids):
        payload = {"model": model, "prompt": ids, "max_tokens": 1,
                   "temperature": 0, "stream": True}
        t0 = time.perf_counter()
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            async for line in resp.content:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    return time.perf_counter() - t0
        return time.perf_counter() - t0

    async def drive(ids_list, concurrency):
        conn = aiohttp.TCPConnector(limit=concurrency)
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
            return await asyncio.gather(*[one(s, ids) for ids in ids_list])

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

    def run_config(label, *, dexa, policy):
        cmd = ["vllm", "serve", model, "--enforce-eager", "--port", str(port),
               "--gpu-memory-utilization", "0.85", "--max-model-len", str(ctx_len + 64),
               "--max-num-batched-tokens", str(ctx_len + 64),  # single-step prefill -> save works
               "--no-enable-prefix-caching"]
        if dexa:
            cmd += ["--kv-transfer-config", json.dumps({
                "kv_connector": "DexaConnector",
                "kv_connector_module_path": "dexa.engine.vllm_connector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "dexa_store_root": f"/tmp/kv_{label}",
                    "dexa_load_policy": policy,
                    # crossover 8192: idle -> don't load ctx<8192; saturated (floor 0.1)
                    # -> ~819 -> load. So adaptive loads THIS ctx only under contention.
                    "dexa_min_load_tokens": "8192",
                },
            })]
        print(f"\n{'='*70}\n[{label}] {' '.join(cmd[:5])} ... policy={policy if dexa else '-'}\n{'='*70}", flush=True)
        server = subprocess.Popen(cmd)
        try:
            if not wait_ready():
                print(f"[{label}] server not ready"); return None
            # populate (dexa): send each once, sequentially, so the connector saves.
            if dexa:
                asyncio.run(drive(contexts, concurrency=1))
                time.sleep(2)
            # measure: all N contexts fired at once -> GPU contention.
            ttfts = asyncio.run(drive(contexts, concurrency=n_sessions))
            ms = [t * 1000 for t in ttfts]
            print(f"[{label}] concurrent TTFT (n={len(ms)}): "
                  f"p50={pct(ms,50):.0f}ms p99={pct(ms,99):.0f}ms mean={sum(ms)/len(ms):.0f}ms "
                  f"max={max(ms):.0f}ms", flush=True)
            return {"label": label, "p50": pct(ms, 50), "p99": pct(ms, 99), "mean": sum(ms)/len(ms)}
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()
            time.sleep(5)

    subprocess.run(["python", "-c", "import torch;print('gpu',torch.cuda.get_device_name(0))"])
    results = [
        run_config("vanilla", dexa=False, policy=None),
        run_config("dexa_always", dexa=True, policy="always"),
        run_config("dexa_adaptive", dexa=True, policy="adaptive"),
    ]
    print("\n" + "=" * 70)
    print(f"CONTENTION @ ctx={ctx_len}, {n_sessions} concurrent sessions ({model}, {GPU})")
    for r in results:
        if r:
            print(f"  {r['label']:14s}  p50={r['p50']:8.0f}ms  p99={r['p99']:8.0f}ms  mean={r['mean']:8.0f}ms")
    print("=" * 70)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", ctx_len: int = 3072, n_sessions: int = 24) -> None:
    print(f"contention benchmark on {GPU}: {model}, ctx={ctx_len}, sessions={n_sessions}")
    bench.remote(model, ctx_len, n_sessions)
