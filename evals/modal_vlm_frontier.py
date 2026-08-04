"""Multimodal cost/performance frontier: how much does cutting visual tokens save,
and does accuracy hold enough to move a workload?

Thesis (execution-owned, not a gateway): visual tokens dominate VLM inference cost — a
chart/doc image is ~1k+ tokens the LLM must prefill and attend over every step — and
they're highly redundant. Reducing them cuts prefill AND decode AND memory. That lever
lives inside the forward pass, so OpenRouter-style routing over commodity APIs can't
offer it. The question that decides the thesis: is the cost cut LARGE at ~flat accuracy?

We sweep the visual-token budget (via input resolution — the simplest deployable knob)
on ChartQA (real chart images, short checkable answers) and measure, per budget:
accuracy (relaxed match), GPU-seconds/image, and avg prompt tokens. A steep frontier
(big cost cut, small accuracy drop) = a noticeable, move-justifying gain. A flat one
(accuracy collapses immediately) kills the thesis for one Modal run.

    modal run evals/modal_vlm_frontier.py
    modal run evals/modal_vlm_frontier.py --n 400
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "qwen-vl-utils", "datasets", "pillow", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-vlm-frontier")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

BUDGETS = [1536, 1024, 768, 512, 384]   # image long-side px -> controls visual token count


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, n: int) -> None:
    import base64
    from io import BytesIO
    from time import perf_counter

    from datasets import load_dataset
    from PIL import Image
    from vllm import LLM, SamplingParams

    # High-res DOCUMENT images: here visual tokens run to thousands and dominate cost,
    # so reducing them actually exercises the lever (ChartQA charts are too small).
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    if n > 0:
        ds = ds.select(range(min(n, len(ds))))
    rows = [(r["image"].convert("RGB"), r["question"], list(r["answers"])) for r in ds]
    print(f"[info] {len(rows)} DocVQA examples", flush=True)

    llm = LLM(model=model, max_model_len=16384, gpu_memory_utilization=0.9,
              enforce_eager=True, limit_mm_per_prompt={"image": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    def resize(img, long_side):
        w, h = img.size
        s = long_side / max(w, h)
        if s >= 1:
            return img
        return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)

    def to_url(img):
        buf = BytesIO(); img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    import re

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    def match(pred, golds):
        p = norm(pred)
        return any(norm(g) == p or (len(norm(g)) > 2 and norm(g) in p) for g in golds)

    def build(B):
        return [[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": to_url(resize(img, B))}},
                    {"type": "text", "text": q + " Answer with just the value, no words."}]}]
                for img, q, _a in rows]

    llm.chat(build(512)[:8], sp, use_tqdm=False)   # WARMUP so the first budget isn't skewed

    results = []
    for B in BUDGETS:
        msgs = build(B)
        t = perf_counter()
        outs = llm.chat(msgs, sp, use_tqdm=True)
        dt = perf_counter() - t
        correct = sum(match(o.outputs[0].text, a) for o, (_i, _q, a) in zip(outs, rows))
        toks = sum(len(o.prompt_token_ids) for o in outs) / len(outs)
        acc = correct / len(rows)
        results.append((B, acc, dt / len(rows), toks, len(rows) / dt))
        print(f"[budget {B:>4}px] acc={acc:.3f}  {len(rows)/dt:6.1f} img/s  "
              f"avg_prompt_tok={toks:6.0f}  ({dt:.1f}s total)", flush=True)
        if B == BUDGETS[0]:
            print(f"   sample: q={rows[0][1][:60]!r} pred={outs[0].outputs[0].text.strip()!r} "
                  f"gold={rows[0][2]!r}", flush=True)

    base = results[0]  # highest-res, warm
    print("\n" + "=" * 80)
    print(f"VLM COST/PERF FRONTIER — {model.split('/')[-1]} ({GPU}), DocVQA, {len(rows)} imgs")
    print("=" * 80)
    print(f"  {'budget':>8} {'accuracy':>9} {'img/s':>8} {'prompt tok':>11} "
          f"{'throughput x':>13} {'Δacc':>7}")
    for B, acc, ms, toks, ips in results:
        print(f"  {B:>6}px {acc:>9.3f} {ips:>8.1f} {toks:>11.0f} "
              f"{ips/base[4]:>12.2f}x {acc-base[1]:>+7.3f}")
    print("=" * 80)
    # best "move-justifying" point: largest throughput gain within 3 acc points of baseline
    ok = [r for r in results if base[1] - r[1] <= 0.03]
    best = max(ok, key=lambda r: r[4]) if ok else base
    print(f"  HEADLINE: at {best[0]}px, {best[4]/base[4]:.1f}x higher throughput "
          f"({base[3]:.0f}->{best[3]:.0f} visual tokens) for {best[1]-base[1]:+.3f} accuracy "
          f"— an inside-the-forward-pass gain no gateway can offer. "
          f"{'Steep frontier => thesis alive' if best[4]/base[4] >= 1.8 else 'Shallow => weak'}.")
    print("=" * 80)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-VL-7B-Instruct", n: int = 200) -> None:
    print(f"VLM cost/perf frontier on {GPU}: {model}")
    run.remote(model, n)
