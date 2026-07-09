"""Independent serving benchmark: `vllm bench serve` (vLLM's own harness) driving a
Dexa-connector server vs a no-cache baseline vs vanilla prefix caching.

This is the `docs/BENCHMARK_PLAN.md` first experiment, run with the credible upstream
harness (not a Dexa-built one). Each config launches a real `vllm serve` and drives it
with `vllm bench serve --dataset-name prefix_repetition` (shared prefixes with varying
suffixes). We use a LONG shared prefix so re-prefill is genuinely expensive — a single
warm GPU is otherwise Dexa's worst case (cheap re-prefill), so a short prefix would
just measure disk overhead.

Configs (prefix caching OFF except the last, so the reuse path is explicit):
  * baseline   — vanilla vLLM, no prefix cache: every request re-prefills the prefix.
  * dexa       — DexaConnector (blob store): first hit per prefix saves, rest load.
  * prefixcache— vanilla vLLM with its in-GPU prefix cache ON (the "why not just
                 prefix caching" reference; in-memory, single-instance, non-portable).

Metric of record: TTFT (mean / P99), which is what prefix reuse targets. Throughput
reported too. Honest read: on one warm instance, `prefixcache` (in-GPU) is the ceiling
and `dexa` (disk load) sits between it and `baseline`; Dexa's real edge is CROSS-
INSTANCE (a cold instance has no local cache — see modal_connector_xinstance.py).

    modal run scripts/modal_bench_serve.py
    modal run scripts/modal_bench_serve.py --model Qwen/Qwen3-0.6B --prefix-len 4096
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_BENCH_GPU", "A10G")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "numpy", "pandas", "datasets")
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("src", "/root/dexa/src")
)

app = modal.App("dexa-bench-serve")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/cache/hf": hf_cache})
def bench(model: str, prefix_len: int, num_prompts: int) -> None:
    import json
    import subprocess
    import time
    import urllib.request

    def wait_ready(port: int, timeout: int = 300) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                return True
            except Exception:
                time.sleep(2)
        return False

    def run_config(label: str, *, dexa: bool, prefix_cache: bool) -> None:
        port = 8000
        cmd = ["vllm", "serve", model, "--enforce-eager", "--port", str(port),
               "--gpu-memory-utilization", "0.55", "--max-model-len", str(prefix_len + 256)]
        cmd += [] if prefix_cache else ["--no-enable-prefix-caching"]
        if dexa:
            cmd += ["--kv-transfer-config", json.dumps({
                "kv_connector": "DexaConnector",
                "kv_connector_module_path": "dexa.engine.vllm_connector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"dexa_store_root": f"/tmp/dexa_{label}"},
            })]
        print(f"\n{'='*70}\n[{label}] launching: {' '.join(cmd[:6])} ... (dexa={dexa}, prefix_cache={prefix_cache})\n{'='*70}", flush=True)
        server = subprocess.Popen(cmd)
        try:
            if not wait_ready(port):
                print(f"[{label}] server did not become ready", flush=True)
                return
            bench_cmd = [
                "vllm", "bench", "serve", "--backend", "vllm", "--model", model,
                "--base-url", f"http://localhost:{port}",
                "--dataset-name", "prefix_repetition",
                "--num-prompts", str(num_prompts),
                "--prefix-repetition-prefix-len", str(prefix_len),
                "--prefix-repetition-suffix-len", "64",
                "--prefix-repetition-num-prefixes", "5",
                "--prefix-repetition-output-len", "32",
                "--percentile-metrics", "ttft,tpot,itl,e2el",
            ]
            r = subprocess.run(bench_cmd, capture_output=True, text=True)
            print(r.stdout[-4000:], flush=True)
            if r.returncode != 0:
                print(f"[{label}] bench stderr:\n{r.stderr[-2000:]}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()
            time.sleep(5)  # let the GPU free before the next config

    subprocess.run(["python", "-c", "import torch;print('torch',torch.__version__,'cuda',torch.cuda.get_device_name(0))"])
    run_config("baseline", dexa=False, prefix_cache=False)
    run_config("dexa", dexa=True, prefix_cache=False)
    run_config("prefixcache", dexa=False, prefix_cache=True)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "facebook/opt-125m", prefix_len: int = 1024, num_prompts: int = 60) -> None:
    print(f"serving benchmark on {GPU}: {model}, prefix_len={prefix_len}, num_prompts={num_prompts}")
    bench.remote(model, prefix_len, num_prompts)
