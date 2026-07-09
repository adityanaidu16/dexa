"""End-to-end validation of DexaConnector through a REAL vLLM engine.

Runs two identical requests against one vLLM with `enable_prefix_caching=False`, so
the DexaConnector store is the ONLY cross-request KV-reuse path:

* request 1 → store miss → normal prefill → **save** the prompt-prefix KV (exercises
  the save-side shims: `_block_ids`, `_split_kv_layer`, `_spec`, `_save_geometry`);
* request 2 → store **HIT** → **load** the KV, skip re-prefill (exercises the
  scheduler match + `_layer_kv_tensors` load path).

Success = a KV file is written after request 1, the "[dexa] store HIT" + "[dexa]
saved KV" logs appear, and both outputs are identical. This is the "one flag in
vLLM" promise made real (docs/CONNECTOR_COMPLETION.md).

    modal run scripts/modal_connector_serve.py
    modal run scripts/modal_connector_serve.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_SERVE_GPU", "A10G")

# Same CUDA-devel template the probe validated (a real vLLM engine needs nvcc).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "numpy")
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("src", "/root/dexa/src")
)

app = modal.App("dexa-connector-serve")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=1800, volumes={"/cache/hf": hf_cache})
def serve_reuse(model: str) -> None:
    import glob

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    store = "/tmp/dexa_kv"
    kt = KVTransferConfig(
        kv_connector="DexaConnector",
        kv_connector_module_path="dexa.engine.vllm_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"dexa_store_root": store},
    )
    # prefix caching OFF -> the connector store is the only cross-request reuse path.
    llm = LLM(model=model, kv_transfer_config=kt, enforce_eager=True,
              gpu_memory_utilization=0.6, max_model_len=2048,
              enable_prefix_caching=False)
    sp = SamplingParams(max_tokens=16, temperature=0.0)
    # >1 block (block_size 16): KV reuse is block-granular, so a sub-block prompt has
    # nothing to reuse. This tokenizes to ~40 tokens (>=2 full blocks).
    prompt = (
        "The quarterly review covered revenue, churn, and the migration plan in "
        "detail, and the team noted that the new pipeline reduced latency while the "
        "storage tier absorbed the additional load. In summary, the capital of France is"
    )

    print("\n=== REQUEST 1 (expect: miss -> prefill -> save) ===", flush=True)
    o1 = llm.generate([prompt], sp)[0].outputs[0].text
    print(f"out1: {o1!r}", flush=True)
    files = [os.path.basename(f) for f in glob.glob(store + "/*")]
    print(f"store files after request 1: {files}", flush=True)

    print("\n=== REQUEST 2 (expect: store HIT -> load -> skip prefill) ===", flush=True)
    o2 = llm.generate([prompt], sp)[0].outputs[0].text
    print(f"out2: {o2!r}", flush=True)

    saved = len(files) > 0
    identical = o1 == o2
    print("\n" + "=" * 60)
    print(f"RESULT: saved={saved}  identical_output={identical}")
    print("(look for '[dexa] saved KV' after req 1 and '[dexa] store HIT' before req 2)")
    print("=" * 60, flush=True)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "facebook/opt-125m") -> None:
    print(f"validating DexaConnector end-to-end on {GPU} with {model} ...")
    serve_reuse.remote(model)
