"""The real Dexa backend — Qwen2.5-VL-7B tuned for computer-use agents.

This is the model server the gateway forwards to. It differs from a stock `vllm serve` in the
things that matter for agents:

  * Prefix caching ON. An agent resends a large, static prefix every step — system prompt,
    tool schema, few-shot — so caching the shared prefix is a real, compounding latency/cost
    win on this workload (not a novelty; just correctly turned on).
  * A screenshot-appropriate visual-token budget (max_pixels). Screenshots are wide and
    text-dense; DEXA_MAX_PIXELS sets the point where UI text stays legible without paying for
    tokens the task doesn't use. ~1.05MP is a sane default for 1280x800-class screens.
  * Multiple images per prompt allowed, so an agent can pass a couple of history frames.
  * Served under the name `dexa-cua-vlm`, matching the gateway.

Two ways to run it — same flags, so cost/quality is identical whichever you pick:

    modal deploy serve/cua_backend.py            # Dexa-hosted (or your own Modal account)
    # BYOC: on any A100/H100-80GB box in your own cloud:
    DEXA_MAX_PIXELS=1050000 serve/backend_launch.sh
    # or docker: see dexa_platform/docker-compose.byoc.yml

The gateway reaches it at $URL/v1/chat/completions — set DEXA_BACKEND_URL to that.
"""

import os
import subprocess

import modal

MODEL = os.environ.get("DEXA_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_PIXELS = int(os.environ.get("DEXA_MAX_PIXELS", "1050000"))  # screenshot-tuned budget
MAX_MODEL_LEN = int(os.environ.get("DEXA_MAX_MODEL_LEN", "16384"))
MAX_IMAGES = int(os.environ.get("DEXA_MAX_IMAGES", "3"))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "qwen-vl-utils")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-cua-backend")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


def _vllm_cmd() -> list[str]:
    """The single source of truth for how the backend is launched — shared by Modal and BYOC."""
    return [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", "dexa-cua-vlm",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--limit-mm-per-prompt", f'{{"image": {MAX_IMAGES}}}',
        "--mm-processor-kwargs", f'{{"max_pixels": {MAX_PIXELS}}}',
    ]


@app.function(image=image, gpu="A100-80GB", volumes={"/cache/hf": hf_cache},
              timeout=3600, scaledown_window=300, min_containers=0)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    subprocess.Popen(_vllm_cmd())


if __name__ == "__main__":
    # `python serve/cua_backend.py` prints the exact BYOC launch command.
    print(" ".join(_vllm_cmd()))
