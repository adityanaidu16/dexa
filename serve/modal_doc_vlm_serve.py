"""MVP: an OpenAI-compatible document-VLM endpoint (the product surface).

A customer points their OpenAI client's base_url here and keeps their code — same
chat/completions API, image_url content, everything — but requests are served by
Qwen2.5-VL-7B at the cost-optimal visual-token budget (max_pixels capped to the
~1024px operating point that measured 0.925 on DocVQA). That's the whole pitch:
one line changed, ~40x cheaper than GPT-4o, higher document accuracy.

    modal deploy serve/modal_doc_vlm_serve.py     # stable URL, scales to zero when idle
    modal serve  serve/modal_doc_vlm_serve.py     # ephemeral dev URL
    modal app stop dexa-doc-vlm                    # tear it down

The endpoint is vLLM's own OpenAI-compatible server; we just launch it with the
document-tuned config. Served model name is "doc-vlm".
"""

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 1048576  # ~1024x1024 — the measured cost/accuracy sweet spot for documents

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "qwen-vl-utils")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-doc-vlm")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", volumes={"/cache/hf": hf_cache},
              timeout=3600, scaledown_window=300, min_containers=0)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", "doc-vlm",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.9",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-kwargs", f'{{"max_pixels": {MAX_PIXELS}}}',
    ]
    subprocess.Popen(cmd)
