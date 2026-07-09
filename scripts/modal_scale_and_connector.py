"""Turnkey Modal run: the 32k/64k persist scaling curve + the vLLM connector check.

Two things this repo can only *validate* on a real GPU/vLLM box, in one command:

1. **Scaling curve to 32k/64k** — `benchmarks/persist_demo.py bench` on a real 8B,
   measuring cold re-prefill vs Dexa state-reload (TTFT), the speedup, lossless
   equality, and persisted state size at each length. The existing `RESULTS.md`
   curve tops out at SmolLM2/CPU and ~8k; this extends it to the lengths that make
   the wedge (compute >> I/O grows with context) actually bite.
2. **vLLM connector conformance** — `benchmarks/vllm_connector_check.py` against
   the *installed* vLLM: subclass check + per-method V1 signature diff (tier 1),
   and optionally loading `DexaConnector` through vLLM's registry and generating
   (tier 2, `--serve`).

Modal is serverless (no capacity lottery, per-second billing) and caches the
model in a persisted HF volume so it downloads once.

Prereqs (once):  pip install modal  &&  modal token new

Run (from the repo root):
    modal run scripts/modal_scale_and_connector.py                       # both halves
    modal run scripts/modal_scale_and_connector.py --only persist        # just the curve
    modal run scripts/modal_scale_and_connector.py --lengths 8000,32000,64000
    modal run scripts/modal_scale_and_connector.py --only connector --serve
    DEXA_PERSIST_GPU=H100 modal run scripts/modal_scale_and_connector.py  # pick the GPU

Sizing note. The persist bench upcasts KV to fp32 for the portable state, so an 8B
at 64k tokens holds ~16 GB of state (and ~24 GB on the device: 16 GB bf16 weights +
8 GB bf16 KV). Default GPU is A100-80GB for headroom; drop to A100-40GB only if you
cap `--lengths` at 32000.
"""

from __future__ import annotations

import os

import modal

# GPUs are env-selectable so you don't edit the file. The persist curve at 64k
# wants an 80GB card; the connector check only needs vLLM to import (tier 1) or a
# tiny model (tier 2), so a small card is plenty.
GPU_PERSIST = os.environ.get("DEXA_PERSIST_GPU", "A100-80GB")
GPU_CONNECTOR = os.environ.get("DEXA_CONNECTOR_GPU", "A10G")

_ENV = {"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"}


def _with_repo(image: "modal.Image") -> "modal.Image":
    # mount only what the benchmarks import — avoids shipping .venv/.git.
    return (
        image.env(_ENV)
        .add_local_dir("src", "/root/dexa/src")
        .add_local_dir("benchmarks", "/root/dexa/benchmarks")
    )


# torch/transformers only — the persist bench uses HFBackend, not vLLM.
torch_image = _with_repo(
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch", "transformers>=4.44", "accelerate", "numpy",
        "safetensors", "sentencepiece",
    )
)

# vLLM present — pulls torch/transformers transitively; this is the image that
# makes tiers 1-2 of the connector check meaningful.
vllm_image = _with_repo(
    modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "numpy")
)

app = modal.App("dexa-scale-connector")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

_REPO = "/root/dexa"
_OUT = f"{_REPO}/benchmarks/out"


def _run(title: str, args: list[str]) -> int:
    import subprocess
    import sys

    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n$ {' '.join(args)}", flush=True)
    return subprocess.run([sys.executable, *args], cwd=_REPO).returncode


@app.function(image=torch_image, gpu=GPU_PERSIST, timeout=3600,
              volumes={"/cache/hf": hf_cache})
def persist_scaling(model: str, lengths: str) -> None:
    """The scaling curve: cold re-prefill vs Dexa state-reload across lengths."""
    _run("GPU sanity",
         ["-c", "import torch;print('torch',torch.__version__,'cuda',"
          "torch.cuda.is_available(),torch.cuda.get_device_name(0) "
          "if torch.cuda.is_available() else '-')"])
    _run(f"Persist scaling curve — {model} @ lengths={lengths} (cold vs resume)",
         [f"{_REPO}/benchmarks/persist_demo.py", "bench", "--model", model,
          "--device", "cuda", "--lengths", lengths,
          "--store-dir", "/tmp/dexa_sessions", "--out-dir", _OUT])
    # echo the machine-readable curve into the streamed logs.
    _run("Curve JSON", ["-c",
         f"import pathlib;print(pathlib.Path('{_OUT}/persist.json').read_text())"])
    hf_cache.commit()


@app.function(image=vllm_image, gpu=GPU_CONNECTOR, timeout=1800,
              volumes={"/cache/hf": hf_cache})
def connector_check(serve: bool, model: str) -> None:
    """Validate DexaConnector against the installed vLLM (tier 1, + tier 2 if serve)."""
    _run("vLLM present?", ["-c",
         "import dexa.engine.vllm_connector as v;"
         "print('vllm_available',v.vllm_available(),'version',v.vllm_version())"])
    args = [f"{_REPO}/benchmarks/vllm_connector_check.py",
            "--store-dir", "/tmp/dexa_connector_check",
            "--out", f"{_OUT}/connector_check.json"]
    if serve:
        args += ["--serve", "--model", model]
    rc = _run("Connector conformance (tier 0 numpy, tier 1 signatures"
              + (", tier 2 engine)" if serve else ")"), args)
    print(f"\nconnector_check exit code: {rc} "
          f"(0 = tiers that ran passed; tier 2 stopping at a documented shim is "
          f"reported, not a hard fail)")
    hf_cache.commit()


@app.local_entrypoint()
def main(
    model: str = "unsloth/Llama-3.1-8B-Instruct",
    lengths: str = "8000,16000,32000,64000",
    connector_model: str = "facebook/opt-125m",
    serve: bool = False,
    only: str = "both",
) -> None:
    """``only``: ``both`` (default), ``persist``, or ``connector``."""
    if only not in ("both", "persist", "connector"):
        raise SystemExit(f"--only must be both|persist|connector, got {only!r}")
    print(f"Dexa scale+connector on Modal: only={only} model={model} "
          f"lengths={lengths} serve={serve}")
    if only in ("both", "persist"):
        print(f"  -> persist scaling on {GPU_PERSIST}")
        persist_scaling.remote(model, lengths)
    if only in ("both", "connector"):
        print(f"  -> connector check on {GPU_CONNECTOR}")
        connector_check.remote(serve, connector_model)
