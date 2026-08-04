"""Measured head-to-head: incumbent vision APIs on the SAME DocVQA pages as our VLM.

Turns the cost/quality artifact's "estimate" chips into measured numbers. Runs GPT-4o and
GPT-4o-mini on the identical 200 DocVQA validation pages (same set our Qwen2.5-VL run used),
scores with the same relaxed match, and computes REAL cost per 1,000 pages from the token
usage the API returns. No secrets in this file — the API key is injected as a Modal secret
from the caller's environment at run time.

    OPENAI_API_KEY=sk-... modal run evals/modal_incumbent_docvqa.py
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("openai", "datasets", "pillow", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-incumbent-docvqa")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

# Published list prices (USD / 1M tokens), early 2026 — adjust if they move.
PRICING = {
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}


@app.function(image=image, timeout=3600, volumes={"/cache/hf": hf_cache},
              secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY"])])
def run(models: list[str], n: int) -> None:
    import base64
    import re
    from concurrent.futures import ThreadPoolExecutor
    from io import BytesIO

    from datasets import load_dataset
    from openai import OpenAI

    client = OpenAI()
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation").select(range(n))
    rows = [(r["image"].convert("RGB"), r["question"], list(r["answers"])) for r in ds]
    print(f"[info] {len(rows)} DocVQA pages (same set as the Qwen run)", flush=True)

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    def match(pred, golds):
        p = norm(pred)
        return any(norm(g) == p or (len(norm(g)) > 2 and norm(g) in p) for g in golds)

    def to_url(img):
        w, h = img.size
        s = 2048 / max(w, h)
        if s < 1:
            img = img.resize((int(w*s), int(h*s)))
        buf = BytesIO(); img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def ask(model, img, q, gold):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=32, temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": q + " Answer with just the value, no words."},
                    {"type": "image_url",
                     "image_url": {"url": to_url(img), "detail": "high"}}]}])
            txt = r.choices[0].message.content or ""
            return match(txt, gold), r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            print(f"  [err] {model}: {e}", flush=True)
            return None

    results = {}
    for model in models:
        with ThreadPoolExecutor(max_workers=12) as ex:
            out = list(ex.map(lambda r: ask(model, *r), rows))
        ok = [o for o in out if o is not None]
        correct = sum(c for c, _p, _o in ok)
        pin = sum(p for _c, p, _o in ok)
        pout = sum(o for _c, _p, o in ok)
        pr = PRICING.get(model, {"in": 0, "out": 0})
        cost_1k = (pin * pr["in"] + pout * pr["out"]) / 1e6 / len(ok) * 1000
        results[model] = (correct / len(ok), pin / len(ok), pout / len(ok), cost_1k, len(ok))
        print(f"[{model}] acc={correct/len(ok):.3f}  in_tok={pin/len(ok):.0f}  "
              f"out_tok={pout/len(ok):.0f}  ${cost_1k:.3f}/1k pages  (n={len(ok)})", flush=True)

    print("\n" + "=" * 74)
    print(f"INCUMBENT HEAD-TO-HEAD — DocVQA ({len(rows)} pages), measured")
    print("=" * 74)
    print(f"  {'model':16} {'DocVQA acc':>11} {'$ / 1k pages':>14} {'avg in tok':>11}")
    for m, (acc, ti, to, c, k) in results.items():
        print(f"  {m:16} {acc:>11.3f} {c:>13.3f}  {ti:>10.0f}")
    print(f"  {'Us (Qwen2.5-VL-7B)':16} {0.925:>11.3f} {'0.02-0.06':>14} {1017:>11.0f}")
    print("=" * 74)
    print("  (Our row from the measured Qwen run; cost = rented-A100 serving, not tokens.)")
    print("=" * 74)


@app.local_entrypoint()
def main(models: str = "gpt-4o,gpt-4o-mini", n: int = 200) -> None:
    print("incumbent DocVQA head-to-head")
    run.remote([m.strip() for m in models.split(",")], n)
