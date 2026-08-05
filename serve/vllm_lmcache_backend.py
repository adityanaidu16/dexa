"""Stateful backend: vLLM + LMCache as a persistent OpenAI-compatible endpoint.

This is the serving tier behind the session service (dexa_platform/sessions/). vLLM serves
the model; LMCache offloads KV to CPU (and optionally NVMe) and restores it on a prefix match
— so when a session resends its growing context, the shared prefix is restored instead of
re-prefilled. Proven in evals/modal_lmcache_restore.py; this makes it a stable URL.

    modal deploy serve/vllm_lmcache_backend.py
    # -> point the session service at it:
    DEXA_SESSION_BACKEND=https://<url> uvicorn dexa_platform.sessions.service:app --port 8070

vLLM's own prefix caching is left OFF so LMCache is the reuse path (matching the measured run).
Served model name: dexa-cua-vlm.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "lmcache")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({
        "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1", "VLLM_USE_V1": "1",
        # LMCache: CPU RAM tier (the proven config). NVMe/disk tier can be added with
        # LMCACHE_LOCAL_DISK=file:///cache/lmcache + LMCACHE_MAX_LOCAL_DISK_SIZE once validated.
        "LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "40",
        "LMCACHE_CHUNK_SIZE": "256",
    })
)

app = modal.App("dexa-stateful-backend")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", volumes={"/cache/hf": hf_cache},
              timeout=3600, scaledown_window=300, min_containers=0)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", "dexa-cua-vlm",
        "--max-model-len", "20000",
        "--gpu-memory-utilization", "0.80",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
    ]
    subprocess.Popen(cmd)
