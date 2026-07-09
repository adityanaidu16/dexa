"""Run Dexa's Phase-1 validation on Modal — real trained 8B, on-demand GPU.

Modal is serverless, so there is no "no instances available" lottery: it schedules
your run onto free capacity and bills per second (a ~10-min run on an A100 is well
under a dollar). This mounts the repo, runs the two benchmarks that need a *trained*
model — incremental recompute and selective (HKVD) recompute — and streams the
output back to your terminal. A persisted HF-cache volume means the ~16 GB model
downloads only once.

Prereqs (once):  pip install modal  &&  modal token new

Run (from the repo root):
    modal run scripts/modal_validate.py
    modal run scripts/modal_validate.py --model unsloth/Llama-3.1-8B-Instruct --reps 12
    DEXA_GPU=H100 modal run scripts/modal_validate.py          # pick the GPU

What to look for. On random weights the selective-recompute curve is flat and only
the *ordering* (HKVD < random < recent) holds. On a real trained model the
KV-deviation distribution is far peakier — this run is where CacheBlend's "~10-15%
recompute recovers most of the quality" either shows up or doesn't.
"""

from __future__ import annotations

import os

import modal

# GPU is env-selectable so you don't have to edit the file: A100 (40GB) is plenty
# for an 8B at these context lengths; use A100-80GB / H100 for the long-context work.
GPU = os.environ.get("DEXA_GPU", "A100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=4.44", "accelerate", "numpy",
        "safetensors", "sentencepiece",
    )
    # xet transfer stalls in some envs (see scripts/runpod_bootstrap.sh); disable it.
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    # mount only what's needed — avoids shipping .venv/.git.
    .add_local_dir("src", "/root/dexa/src")
    .add_local_dir("benchmarks", "/root/dexa/benchmarks")
)

app = modal.App("dexa-validate")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/cache/hf": hf_cache})
def validate(model: str, reps: int, steps: int, scale: bool, lengths: str) -> None:
    import subprocess
    import sys

    B = "/root/dexa/benchmarks"

    def run(title, args):
        print(f"\n{'='*70}\n{title}\n{'='*70}\n$ {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, *args], check=False)

    # sanity: confirm we actually have a GPU.
    subprocess.run([sys.executable, "-c",
                    "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),"
                    "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"], check=False)

    if scale:
        # focused scaling artifact: speedup vs context length (skips the rest).
        run("On-device incremental — SCALING: speedup vs context length (O(delta) vs O(context))",
            [f"{B}/ondevice_incremental_bench.py", "--scaling", "--lengths", lengths,
             "--device", "cuda", "--model", model])
        hf_cache.commit()
        return

    run("Layer B — ON-DEVICE incremental recompute (wall-time: full vs numpy vs on-device)",
        [f"{B}/ondevice_incremental_bench.py", "--steps", str(steps),
         "--device", "cuda", "--model", model])

    run("Layer C — selective (HKVD) recompute vs recency/random (the magnitude test)",
        [f"{B}/selective_recompute_bench.py", "--device", "cuda",
         "--reps", str(reps), "--model", model])

    hf_cache.commit()   # persist the downloaded model for next time


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", reps: int = 10, steps: int = 20,
         scale: bool = False, lengths: str = "512,1024,2048,4096,8192,16384,32768") -> None:
    mode = "scaling sweep" if scale else "full Phase-1 validation"
    print(f"running Dexa {mode} on Modal ({GPU}) with {model} ...")
    validate.remote(model, reps, steps, scale, lengths)
