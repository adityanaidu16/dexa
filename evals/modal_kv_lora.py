"""Is a base model's KV cache reusable by a LoRA-adapted version of the same base?

Full fine-tunes broke KV interchange (base<->instruct diverged in 2-5 tokens). But LoRA
is different: the base weights are FROZEN and only a small low-rank delta is added. That's
the practical multi-tenant case — "prefill once with the base, serve many adapters" — and
the one place KV reuse across weights might actually hold.

We test it as a controlled sweep rather than one adapter: synthesize a LoRA-style low-rank
delta on the standard target projections (q/k/v/o/gate/up/down) at a controlled *relative*
strength s = ||delta||_F / ||W||_F, and sweep s. For each strength we prefill the test
context with the BASE, hand that KV to the ADAPTED model, and measure how much of the
adapted model's own greedy output survives. s=0 is identity (100% by construction); as s
grows the adapter diverges from the base and we watch KV reuse break.

  reference : adapted model generates from ITS OWN KV        (ceiling)
  reuse     : adapted model generates from the BASE's KV     (what "prefill once" buys)

    modal run evals/modal_kv_lora.py

Caveat: a random low-rank delta spreads energy across all directions, so for a given norm
it likely perturbs activations MORE than a task-trained LoRA (which concentrates in a few
directions). So the strength at which reuse breaks here is a conservative lower bound —
a real trained adapter of the same norm should reuse at least this well, probably better.
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

app = modal.App("dexa-kv-lora")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

TEST = (
    "Bees communicate the direction and distance of flowers through a waggle dance. The "
    "Silk Road carried not only goods but ideas, religions, and diseases across Eurasia. "
    "Superconductors expel magnetic fields and carry current without resistance below a "
    "critical temperature. The Library of Alexandria collected scrolls from across the "
    "ancient world before its gradual decline. Tectonic plates drift a few centimeters a "
    "year, reshaping continents over millions of years. The invention of the stirrup "
    "transformed mounted warfare in medieval Europe. The abacus enabled rapid arithmetic "
    "millennia before electronic calculators existed. Ocean currents redistribute heat "
    "from the equator toward the poles, moderating coastal climates. "
)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
STRENGTHS = [0.0, 0.01, 0.03, 0.10, 0.30]


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/cache/hf": hf_cache})
def run(model_id: str, rank: int, k_gen: int) -> None:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model_id)

    def load():
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(dev).eval()

    def to_legacy(c):
        return c.to_legacy_cache() if hasattr(c, "to_legacy_cache") else c

    @torch.no_grad()
    def prefill(model, ids):
        return to_legacy(model(ids, use_cache=True).past_key_values)

    @torch.no_grad()
    def continue_greedy(model, legacy, last, k):
        cache = DynamicCache.from_legacy_cache(tuple(legacy))
        toks, cur = [], last
        for _ in range(k):
            out = model(cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(cur))
        return toks

    def agree(a, ref):
        m = sum(1 for x, y in zip(a, ref) if x == y)
        div = next((i for i, (x, y) in enumerate(zip(a, ref)) if x != y), len(ref))
        return m / len(ref), div

    def cos(x, y):
        return torch.nn.functional.cosine_similarity(
            x.flatten().float(), y.flatten().float(), dim=0).item()

    base = load()      # frozen base — source of the shared KV, never modified
    lora = load()      # gets base weights + a low-rank delta each sweep step

    # fixed random low-rank factors per target Linear; recompute delta per strength
    torch.manual_seed(0)
    factors, base_w = {}, {}
    for name, mod in base.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith(TARGETS):
            W = mod.weight
            out, inn = W.shape
            A = torch.randn(rank, inn, device=dev, dtype=torch.float32) / (inn ** 0.5)
            B = torch.randn(out, rank, device=dev, dtype=torch.float32) / (rank ** 0.5)
            raw = B @ A
            factors[name] = (A, B, W.float().norm().item(), raw.norm().item())
            base_w[name] = W.detach()

    def set_strength(s):
        with torch.no_grad():
            for name, mod in lora.named_modules():
                if name in factors:
                    A, B, wn, rn = factors[name]
                    delta = (B @ A) * (s * wn / rn) if s > 0 else 0.0
                    mod.weight.data = (base_w[name].float() + delta).to(mod.weight.dtype)

    ids = tok(TEST, return_tensors="pt").input_ids[:, :256].to(dev)
    prefix, last = ids[:, :-1], ids[:, -1:]
    base_kv = prefill(base, prefix)

    print("\n" + "=" * 78)
    print(f"LoRA-over-shared-base KV reuse — {model_id.split('/')[-1]}, rank={rank}")
    print("=" * 78)
    print(f"  {'strength':>9} {'KV cosine':>11} {'reuse agree':>12} {'first diverge':>14}")
    rows = []
    for s in STRENGTHS:
        set_strength(s)
        lora_kv = prefill(lora, prefix)
        ref = continue_greedy(lora, lora_kv, last, k_gen)          # adapted from own KV
        reuse = continue_greedy(lora, base_kv, last, k_gen)        # adapted from base KV
        frac, div = agree(reuse, ref)
        kcos = sum(cos(lk[0], bk[0]) for lk, bk in zip(lora_kv, base_kv)) / len(lora_kv)
        rows.append((s, kcos, frac, div))
        print(f"  {s:>9.2f} {kcos:>11.3f} {frac*100:>11.1f}% {div:>12}/{k_gen}", flush=True)
    print("=" * 78)
    # find the largest strength that still fully reuses
    ok = [s for s, _kc, f, _d in rows if f >= 0.95]
    hi = max(ok) if ok else 0.0
    print(f"  base KV stays fully reusable up to ~{hi:.0%} relative weight change "
          f"(rank-{rank} random delta; a trained LoRA of equal norm should do at least "
          f"this well). Reference: a full fine-tune (base<->instruct) breaks immediately.")
    print("=" * 78)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Meta-Llama-3.1-8B", rank: int = 16, k_gen: int = 48) -> None:
    print(f"LoRA-over-shared-base KV reuse sweep on {GPU}")
    run.remote(model, rank, k_gen)
