"""Is the KV cache interchangeable across numeric FORMATS and across model WEIGHTS?

KV tensors are activations produced by a specific model's weights, so two axes differ
fundamentally:

  FORMAT axis  (same weights, different encoding): round-trip the KV through
    bf16 -> fp16 -> fp8_e4m3 -> int8 -> int4 (per-(token,head) scaled), reload, and
    continue GREEDILY. How compact can the KV get before the generated tokens drift?
    This is the "portable KV container" question — expected to mostly work.

  WEIGHTS axis (same architecture, different weights): inject model A's KV into model B
    (base <-> instruct fine-tune of the same 8B) and continue with B. Raw KV encodes A's
    entire forward pass, so this is expected to DEGRADE — the question is how fast, and
    whether a few tokens survive (shared base). The divergence curve is the answer.

Protocol (isolates the shared prefix cleanly): prefill tokens[:-1] with model M_kv to get
the prefix cache; then model M_gen consumes the final token from that cache and greedily
decodes K tokens. Reference = M_kv==M_gen, no round-trip. We report token agreement vs
reference, first-divergence step, first-step logit KL, and (weights axis) per-layer KV
cosine similarity so "cache numerically close" is separated from "generation drifts".

    modal run evals/modal_kv_interchange.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "transformers==4.46.3", "accelerate", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-kv-interchange")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

PARA = (
    "The quarterly review covered revenue, churn, and the roadmap for the data platform. "
    "Engineering flagged latency regressions in the ingestion pipeline and proposed a "
    "migration to a streaming architecture with backpressure and exactly-once delivery. "
    "Legal raised questions about data residency across the EU and APAC regions, and the "
    "team debated whether to shard by tenant or by geography given the compliance burden. "
)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/cache/hf": hf_cache})
def run(instruct_id: str, base_id: str, ctx_len: int, k_gen: int) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(instruct_id)
    ids_full = tok(PARA * 20, add_special_tokens=True, return_tensors="pt").input_ids
    ids = ids_full[:, :ctx_len].to(dev)
    prefix, last = ids[:, :-1], ids[:, -1:]

    def load(mid):
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to(dev)
        m.eval()
        return m

    def qrt(x, mode):
        """Quantize->dequantize a KV tensor [B,H,S,D] along head_dim (per token,head)."""
        if mode == "bf16":
            return x
        if mode == "fp16":
            return x.to(torch.float16).to(x.dtype)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        if mode == "fp8":
            scale = amax / 448.0
            q = (x / scale).to(torch.float8_e4m3fn).to(torch.float32)
            return (q * scale).to(x.dtype)
        qmax = 127.0 if mode == "int8" else 7.0
        scale = amax / qmax
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return (q * scale).to(x.dtype)

    BITS = {"bf16": 16, "fp16": 16, "fp8": 8, "int8": 8, "int4": 4}

    @torch.no_grad()
    def prefill_cache(model):
        out = model(prefix, use_cache=True)
        return out.past_key_values

    def to_legacy(cache):
        # transformers 4.46 may return a DynamicCache or a legacy tuple depending on model
        return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache

    def requantize(cache, mode):
        legacy = to_legacy(cache)
        newl = tuple((qrt(k, mode), qrt(v, mode)) for k, v in legacy)
        return DynamicCache.from_legacy_cache(newl)

    @torch.no_grad()
    def continue_greedy(model, cache, k):
        toks, cur, first_logits = [], last, None
        for i in range(k):
            out = model(cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1]
            if i == 0:
                first_logits = logits.float()
            cur = logits.argmax(-1, keepdim=True)
            toks.append(int(cur))
        return toks, first_logits

    def agree(a, ref):
        m = sum(1 for x, y in zip(a, ref) if x == y)
        div = next((i for i, (x, y) in enumerate(zip(a, ref)) if x != y), len(ref))
        return m / len(ref), div

    print(f"[info] transformers cache; ctx={ctx_len}, k_gen={k_gen}", flush=True)
    instruct = load(instruct_id)
    cache_i = prefill_cache(instruct)
    ref_toks, ref_logits = continue_greedy(instruct, requantize(cache_i, "bf16"), k_gen)

    # ---------------- FORMAT axis ----------------
    print("\n" + "=" * 76)
    print(f"FORMAT AXIS — same weights ({instruct_id.split('/')[-1]}), KV round-tripped")
    print("=" * 76)
    print(f"  {'format':8} {'bytes/tok rel':>13} {'token agree':>12} {'1st diverge':>12} {'step-1 KL':>10}")
    for mode in ["bf16", "fp16", "fp8", "int8", "int4"]:
        toks, logits = continue_greedy(instruct, requantize(cache_i, mode), k_gen)
        frac, div = agree(toks, ref_toks)
        kl = F.kl_div(F.log_softmax(logits, -1), F.softmax(ref_logits, -1),
                      reduction="batchmean").item()
        rel = BITS[mode] / 16.0
        print(f"  {mode:8} {rel:>12.2f}x {frac*100:>11.1f}% {div:>10}/{k_gen} {kl:>10.4f}")
    print("=" * 76)

    # ---------------- WEIGHTS axis ----------------
    try:
        base = load(base_id)
    except Exception as e:
        print(f"\n[weights axis skipped: could not load base {base_id}: {e}")
        hf_cache.commit()
        return

    cache_b = prefill_cache(base)
    base_ref, _ = continue_greedy(base, requantize(cache_b, "bf16"), k_gen)

    # per-layer cosine similarity of the two models' KV over identical tokens
    li, lb = to_legacy(cache_i), to_legacy(cache_b)
    kcos = torch.stack([F.cosine_similarity(ki.flatten(), kb.flatten(), dim=0)
                        for (ki, _), (kb, _) in zip(li, lb)]).mean().item()
    vcos = torch.stack([F.cosine_similarity(vi.flatten(), vb.flatten(), dim=0)
                        for (_, vi), (_, vb) in zip(li, lb)]).mean().item()

    print("\n" + "=" * 76)
    print(f"WEIGHTS AXIS — inject KV across {instruct_id.split('/')[-1]} <-> {base_id.split('/')[-1]}")
    print("=" * 76)
    print(f"  mean KV cosine similarity (instruct vs base, same tokens): K={kcos:.3f} V={vcos:.3f}")
    # base generates from INSTRUCT's KV, vs base's own reference
    bi_toks, _ = continue_greedy(base, requantize(cache_i, "bf16"), k_gen)
    f1, d1 = agree(bi_toks, base_ref)
    # instruct generates from BASE's KV, vs instruct's own reference
    ib_toks, _ = continue_greedy(instruct, requantize(cache_b, "bf16"), k_gen)
    f2, d2 = agree(ib_toks, ref_toks)
    print(f"  base gen  <- instruct KV : agree {f1*100:5.1f}%, diverges at {d1}/{k_gen} (vs base's own)")
    print(f"  instruct gen <- base KV  : agree {f2*100:5.1f}%, diverges at {d2}/{k_gen} (vs instruct's own)")
    print("=" * 76)
    verdict_fmt = "FORMAT interchange works down to int8; int4 drifts"
    verdict_w = ("WEIGHTS interchange BREAKS — raw KV is model-specific"
                 if max(f1, f2) < 0.5 else "WEIGHTS partly survive — investigate")
    print(f"  VERDICT: {verdict_fmt}. {verdict_w}.")
    print("=" * 76)
    hf_cache.commit()


@app.local_entrypoint()
def main(instruct: str = "unsloth/Llama-3.1-8B-Instruct",
         base: str = "unsloth/Meta-Llama-3.1-8B",
         ctx_len: int = 1024, k_gen: int = 48) -> None:
    print(f"KV interchange experiments on {GPU}")
    run.remote(instruct, base, ctx_len, k_gen)
