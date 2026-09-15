"""Modal app for the live agent experiment: one vLLM server (one GPU) serving an open coder model on an
OpenAI-compatible HTTPS endpoint with tool calling. Task sandboxes are created on demand from the official
SWE-bench task images by live_agent_modal.py.

Configuration is read from the environment of the machine running `modal deploy` and shipped to the
container as a Modal secret (container code cannot see the deployer's environment otherwise):
  SPEC_MODEL          HF model id                (default Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)
  SPEC_GPU            Modal GPU spec             (default H100)
  SPEC_VLLM_VERSION   vLLM version installed into the image (default 0.11.0; torch is pinned by vLLM itself)
  SPEC_CUDA_IMAGE     base image                 (default nvidia/cuda:12.8.1-devel-ubuntu22.04)
  SPEC_TOOL_PARSER    vLLM tool-call parser      (default qwen3_coder)
  SPEC_MAX_MODEL_LEN  context length             (default 65536)
  SPEC_MAX_NUM_SEQS   max concurrent sequences   (default 64)
  SPEC_VLLM_API_KEY   bearer key for the public endpoint (default spec-exec-local; the workflow sets a random one)

Deploy:  modal deploy app.py     (prints the endpoint URL)
Stop:    modal app stop spec-exec-vllm
"""
import os, subprocess, time, urllib.request
import modal

MODEL = os.environ.get("SPEC_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
GPU = os.environ.get("SPEC_GPU", "H100")
VLLM_VERSION = os.environ.get("SPEC_VLLM_VERSION", "0.11.0")
CUDA_IMAGE = os.environ.get("SPEC_CUDA_IMAGE", "nvidia/cuda:12.8.1-devel-ubuntu22.04")
CONFIG = {
    "SPEC_MODEL": MODEL,
    "SPEC_TOOL_PARSER": os.environ.get("SPEC_TOOL_PARSER", "qwen3_coder"),
    "SPEC_MAX_MODEL_LEN": os.environ.get("SPEC_MAX_MODEL_LEN", "65536"),
    "SPEC_MAX_NUM_SEQS": os.environ.get("SPEC_MAX_NUM_SEQS", "64"),
    "SPEC_VLLM_API_KEY": os.environ.get("SPEC_VLLM_API_KEY", "spec-exec-local"),
}

app = modal.App("spec-exec-vllm")
hf_cache = modal.Volume.from_name("spec-exec-hf-cache", create_if_missing=True)
# Built the way Modal's own vLLM example does it: a CUDA base with Modal's Python added, then vLLM from PyPI.
# (The vllm/vllm-openai image cannot be used for a Modal Function: Modal cannot detect its Python.)
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "flashinfer-python==0.3.1")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/hf": hf_cache},
    secrets=[modal.Secret.from_dict(CONFIG)],
    timeout=60 * 60 * 6,
    startup_timeout=60 * 30,
    scaledown_window=15 * 60,
    min_containers=0,
)
@modal.concurrent(max_inputs=128)
class VLLM:
    @modal.enter()
    def start(self):
        env = os.environ
        cmd = [
            "vllm", "serve", env["SPEC_MODEL"],
            "--host", "0.0.0.0", "--port", "8000",
            "--max-model-len", env["SPEC_MAX_MODEL_LEN"],
            "--max-num-seqs", env["SPEC_MAX_NUM_SEQS"],
            "--enable-auto-tool-choice", "--tool-call-parser", env["SPEC_TOOL_PARSER"],
            "--api-key", env["SPEC_VLLM_API_KEY"],
            "--gpu-memory-utilization", "0.90",
            "--enable-prefix-caching",
        ]
        self.proc = subprocess.Popen(cmd)
        deadline = time.time() + 20 * 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited early with code {self.proc.returncode}")
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("vLLM did not become healthy in 20 minutes")
        try:
            hf_cache.commit()  # persist freshly downloaded weights for the next cold start
        except Exception:
            pass

    @modal.web_server(8000, startup_timeout=60 * 25)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.proc.terminate()
