"""The moat test: does content-aware IN-MODEL visual-token pruning beat naive resize?

The frontier run showed cutting visual tokens saves ~3x at flat accuracy — but the lever
there was resolution reduction, which a client can do before any API. The defensible,
execution-owned lever is keeping the *informative* visual tokens instead of blurring the
whole image: a selection that needs the vision encoder's per-token signal, so a gateway
can't do it. If content-aware selection is MORE accurate than uniform resize at the same
token budget, execution-owned token reduction is a real gain no gateway can replicate.

At matched retained-token fraction f we compare, on DocVQA:
  resize      : downscale the whole image so vision emits ~f of the tokens (client-doable)
  prune-norm  : full res, keep the f tokens with highest embedding norm (saliency proxy)
  prune-query : full res, keep the f tokens most similar to the question embedding
Pruning drops tokens by zeroing their attention mask. Higher accuracy at matched f for a
prune method => smart selection beats resize => the execution-owned moat is real.

    modal run evals/modal_vlm_moat.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "transformers==4.49.0", "accelerate", "qwen-vl-utils",
                 "datasets", "pillow", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-vlm-moat")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

FRACTIONS = [0.5, 0.25]
FULL_LONGSIDE = 1280


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model_id: str, n: int) -> None:
    import re

    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dev = "cuda"
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    img_tok = model.config.image_token_id

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    rows = [(r["image"].convert("RGB"), r["question"], list(r["answers"])) for r in ds]
    print(f"[info] {len(rows)} DocVQA examples, image_token_id={img_tok}", flush=True)

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    def match(pred, golds):
        p = norm(pred)
        return any(norm(g) == p or (len(norm(g)) > 2 and norm(g) in p) for g in golds)

    def resize(img, long):
        w, h = img.size
        s = long / max(w, h)
        return img if s >= 1 else img.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BICUBIC)

    def build(img, q):
        msg = [{"role": "user", "content": [{"type": "image"},
               {"type": "text", "text": q + " Answer with just the value, no words."}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return proc(text=[text], images=[img], return_tensors="pt").to(dev)

    @torch.no_grad()
    def gen(inputs, attention_mask=None):
        am = attention_mask if attention_mask is not None else inputs["attention_mask"]
        out = model.generate(input_ids=inputs["input_ids"], attention_mask=am,
                             pixel_values=inputs["pixel_values"],
                             image_grid_thw=inputs["image_grid_thw"],
                             max_new_tokens=32, do_sample=False)
        return proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    @torch.no_grad()
    def importance(inputs, q):
        ve = model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])  # [nvis,H]
        norms = ve.float().norm(dim=-1)
        qids = proc.tokenizer(q, return_tensors="pt").input_ids.to(dev)
        qemb = model.get_input_embeddings()(qids)[0].float().mean(0)
        sim = torch.nn.functional.cosine_similarity(ve.float(), qemb[None, :], dim=-1)
        return norms, sim

    def masked(inputs, vis_pos, keep_idx):
        am = inputs["attention_mask"].clone()
        drop = torch.ones(len(vis_pos), dtype=torch.bool, device=dev)
        drop[keep_idx] = False
        am[0, vis_pos[drop]] = 0
        return am

    tally = {("resize", f): 0 for f in FRACTIONS}
    tally.update({(m, f): 0 for m in ("norm", "query") for f in FRACTIONS})
    ref = 0
    ntok_full = []
    for i, (img, q, gold) in enumerate(rows):
        full = build(resize(img, FULL_LONGSIDE), q)
        vis_pos = (full["input_ids"][0] == img_tok).nonzero(as_tuple=True)[0]
        nv = len(vis_pos)
        ntok_full.append(nv)
        ref += match(gen(full), gold)
        norms, sim = importance(full, q)
        if len(norms) != nv:
            print(f"  [warn] visual embeds {len(norms)} != positions {nv}", flush=True)
        for f in FRACTIONS:
            k = max(1, int(f * nv))
            # resize to ~sqrt(f) side (tokens ~ area)
            r = build(resize(img, max(56, int(FULL_LONGSIDE * (f ** 0.5)))), q)
            tally[("resize", f)] += match(gen(r), gold)
            keep_n = torch.topk(norms, k).indices
            tally[("norm", f)] += match(gen(full, masked(full, vis_pos, keep_n)), gold)
            keep_q = torch.topk(sim, k).indices
            tally[("query", f)] += match(gen(full, masked(full, vis_pos, keep_q)), gold)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(rows)}", flush=True)

    N = len(rows)
    avg_tok = sum(ntok_full) / N
    print("\n" + "=" * 80)
    print(f"VLM MOAT TEST — {model_id.split('/')[-1]} ({GPU}), DocVQA, {N} imgs, "
          f"full≈{avg_tok:.0f} visual tokens")
    print("=" * 80)
    print(f"  full-resolution accuracy: {ref/N:.3f}")
    print(f"  {'kept':>6} {'resize':>10} {'prune-norm':>12} {'prune-query':>13} {'best vs resize':>16}")
    for f in FRACTIONS:
        rs, nm, qy = (tally[("resize", f)]/N, tally[("norm", f)]/N, tally[("query", f)]/N)
        best = max(nm, qy)
        print(f"  {int(f*100):>4}% {rs:>10.3f} {nm:>12.3f} {qy:>13.3f} {best-rs:>+15.3f}")
    print("=" * 80)
    f = FRACTIONS[-1]
    rs = tally[("resize", f)]/N
    best = max(tally[("norm", f)], tally[("query", f)])/N
    verdict = ("MOAT REAL — content-aware pruning beats resize at matched budget"
               if best - rs >= 0.02 else
               "NO MOAT — smart pruning does not beat naive resize; edge is only adaptive res")
    print(f"  VERDICT (at {int(f*100)}% tokens): best prune {best:.3f} vs resize {rs:.3f} "
          f"({best-rs:+.3f}). {verdict}.")
    print("=" * 80)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-VL-7B-Instruct", n: int = 80) -> None:
    print(f"VLM moat test on {GPU}: {model}")
    run.remote(model, n)
