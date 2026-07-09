"""Discover vLLM's real V1-connector object shapes on a GPU box.

Loads `DexaProbeConnector` (dexa.engine.vllm_probe_connector) into a real vLLM via
the offline `LLM(...)` API on a tiny model, runs one generation, and prints the
recorded structure of every object the DexaConnector site-shims must handle. This
is the ground-truth discovery step for `docs/CONNECTOR_COMPLETION.md` — implement
the shims against what this dumps, not against guesses.

Cheap: tiny model on a small GPU. Run:
    modal run scripts/modal_connector_probe.py
    modal run scripts/modal_connector_probe.py --model Qwen/Qwen3-0.6B
    DEXA_PROBE_GPU=A10G modal run scripts/modal_connector_probe.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_PROBE_GPU", "A10G")

# A real vLLM ENGINE probes nvcc during KV-cache init, which a runtime-only base
# (debian_slim + pip vllm) lacks. vLLM's own image ships nvcc but Modal can't detect
# its Python. Modal's documented vLLM recipe fixes both: a CUDA *devel* base (nvcc +
# CUDA at /usr/local/cuda) + Modal-managed Python + pip vllm pinned to the version we
# validate against. This is the template image for every real-vLLM-engine run.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "numpy")
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("src", "/root/dexa/src")
)

app = modal.App("dexa-connector-probe")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=1800, volumes={"/cache/hf": hf_cache})
def probe(model: str) -> None:
    import glob
    import json

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"vllm {vllm.__version__} on {GPU}; probing connector object shapes with {model}",
          flush=True)

    kt = KVTransferConfig(
        kv_connector="DexaProbeConnector",
        kv_connector_module_path="dexa.engine.vllm_probe_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"dexa_probe_out": "/tmp/dexa_probe.json"},
    )
    llm = LLM(model=model, kv_transfer_config=kt, enforce_eager=True,
              gpu_memory_utilization=0.6, max_model_len=2048)
    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=16, temperature=0.0))
    print(f"generation ok: {out[0].outputs[0].text!r}", flush=True)

    merged: dict = {}
    for path in sorted(glob.glob("/tmp/dexa_probe.*.json")):
        try:
            with open(path) as f:
                part = json.load(f)
        except Exception as e:
            print(f"(could not read {path}: {e})")
            continue
        tag = path.split("dexa_probe.")[-1].rsplit(".json", 1)[0]
        for k, v in part.items():
            merged.setdefault(k, {})[tag] = v

    print("\n" + "=" * 70)
    print("DEXA PROBE — recorded vLLM connector object shapes")
    print("=" * 70)
    print(json.dumps(merged, indent=2, default=str))
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "facebook/opt-125m") -> None:
    print(f"probing vLLM connector shapes on {GPU} with {model} ...")
    probe.remote(model)
