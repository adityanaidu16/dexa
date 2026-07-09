"""Cross-INSTANCE KV reuse: KV saved by one vLLM process, loaded by another.

This is Dexa's actual pitch (portable across instances), a step beyond the
same-process cross-request reuse in `modal_connector_serve.py`. Two *separate* Modal
containers (= two separate vLLM processes) share a persistent Modal Volume as the
DexaConnector store:

* **Instance A** starts with an empty store, prefills the prompt, and **saves** its KV
  to the volume, then commits.
* **Instance B** is a fresh container (fresh vLLM). On entry the volume already carries
  A's saved KV (proof the state is portable), so B gets a store **HIT**, loads the KV,
  skips the block-aligned prefill, and returns **identical** output.

The two `run_instance.remote(...)` calls run sequentially, so A commits before B
starts. Evidence of cross-instance reuse: B sees A's file on entry
(`files_before` non-empty), B logs `[dexa] store HIT`, and `A.text == B.text`.

    modal run scripts/modal_connector_xinstance.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_XINST_GPU", "A10G")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "numpy")
    .env({"PYTHONPATH": "/root/dexa/src", "HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("src", "/root/dexa/src")
)

app = modal.App("dexa-connector-xinstance")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)
# the shared, portable KV store — this is what carries state ACROSS instances.
kv_store = modal.Volume.from_name("dexa-kv-store", create_if_missing=True)

_PROMPT = (
    "The quarterly review covered revenue, churn, and the migration plan in detail, "
    "and the team noted that the new pipeline reduced latency while the storage tier "
    "absorbed the additional load. In summary, the capital of France is"
)


@app.function(image=image, gpu=GPU, timeout=1800,
              volumes={"/cache/hf": hf_cache, "/kv": kv_store})
def run_instance(model: str, label: str) -> dict:
    import glob

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_store.reload()  # see any KV a prior instance committed
    before = [os.path.basename(f) for f in glob.glob("/kv/*")]
    print(f"\n[{label}] store on entry: {before}", flush=True)

    kt = KVTransferConfig(
        kv_connector="DexaConnector",
        kv_connector_module_path="dexa.engine.vllm_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"dexa_store_root": "/kv"},
    )
    llm = LLM(model=model, kv_transfer_config=kt, enforce_eager=True,
              gpu_memory_utilization=0.6, max_model_len=2048,
              enable_prefix_caching=False)
    text = llm.generate([_PROMPT], SamplingParams(max_tokens=16, temperature=0.0))[0].outputs[0].text

    after = [os.path.basename(f) for f in glob.glob("/kv/*")]
    kv_store.commit()
    print(f"[{label}] out: {text!r}", flush=True)
    print(f"[{label}] store after: {after}", flush=True)
    return {"label": label, "text": text, "files_before": before, "files_after": after}


@app.local_entrypoint()
def main(model: str = "facebook/opt-125m", reset: bool = False) -> None:
    print(f"cross-instance KV reuse on {GPU} with {model} ...")
    a = run_instance.remote(model, "A(save)")
    b = run_instance.remote(model, "B(load)")

    carried = len(b["files_before"]) > 0            # B saw A's KV -> portable state
    a_saved = len(a["files_after"]) > len(a["files_before"])
    identical = a["text"] == b["text"]
    print("\n" + "=" * 64)
    print(f"A(save): before={a['files_before']} after={a['files_after']}")
    print(f"B(load): before={b['files_before']} after={b['files_after']}")
    print(f"CROSS-INSTANCE RESULT: A_saved={a_saved}  "
          f"B_saw_stored_KV={carried}  identical_output={identical}")
    print("(B's container is a separate vLLM process; seeing A's KV on entry + a "
          "'[dexa] store HIT' in B's logs proves portable cross-instance reuse)")
    print("=" * 64)
