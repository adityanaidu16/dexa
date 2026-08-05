"""Delta-perception viability: when a small screen region changes, do the UNCHANGED
regions' vision embeddings stay stable enough to reuse?

The 7x agent-redundancy prize is only capturable if you can reuse computation for the
~86% of the screen that didn't change. That requires the vision encoder to be LOCAL —
a change in one region must NOT perturb the embeddings elsewhere. Qwen2.5-VL uses windowed
vision attention, so it should hold, but that's the load-bearing assumption of the whole
thesis, so we measure it.

Method: capture consecutive agent frames (Playwright drives the CRM app), run Qwen2.5-VL's
vision encoder on each, and for consecutive frames compare per-token embeddings (aligned by
position — same viewport, same grid). If most tokens have cosine ~1.0 (stable) and only a
small fraction shifted — matching the pixel-change fraction — the encoding is local and the
unchanged regions are reusable. If the change propagates (many tokens shift), it isn't.

    modal run evals/agent_redundancy/modal_delta_viability.py
"""

import base64
import pathlib

import modal

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
APP = pathlib.Path(__file__).parent / "app.html"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "transformers==4.49.0", "accelerate", "qwen-vl-utils",
                 "pillow", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)
app = modal.App("dexa-delta-viability")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=1800, volumes={"/cache/hf": hf_cache})
def run(shots_b64: list[str], labels: list[str]) -> None:
    import io

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dev = "cuda"
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(dev).eval()

    shots = [Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB") for s in shots_b64]

    @torch.no_grad()
    def embed(img):
        ip = proc.image_processor(images=[img], return_tensors="pt")
        ve = model.visual(ip["pixel_values"].to(dev, torch.bfloat16),
                          grid_thw=ip["image_grid_thw"].to(dev))
        return ve.float(), ip["image_grid_thw"][0].tolist()

    def pixel_change_frac(a, b, block=28):
        A = np.asarray(a.convert("L"), np.int16); B = np.asarray(b.convert("L"), np.int16)
        h = (min(A.shape[0], B.shape[0]) // block) * block
        w = (min(A.shape[1], B.shape[1]) // block) * block
        d = np.abs(A[:h, :w] - B[:h, :w]).reshape(h // block, block, w // block, block).mean((1, 3))
        return float((d > 6).mean())

    embs = [embed(s) for s in shots]

    print("\n" + "=" * 74)
    print("DELTA-PERCEPTION VIABILITY — Qwen2.5-VL vision embeddings across agent frames")
    print("=" * 74)
    print(f"  {'action':30} {'pixels chg':>11} {'tokens shifted':>15} {'reusable':>10}")
    rel = []
    for i, lab in enumerate(labels):
        e0, g0 = embs[i]; e1, g1 = embs[i + 1]
        if g0 != g1 or e0.shape != e1.shape:
            print(f"  {lab:30}  (grid changed — skipped)")
            continue
        cos = torch.nn.functional.cosine_similarity(e0, e1, dim=-1)  # [n_tokens]
        shifted = float((cos < 0.98).float().mean())     # fraction of tokens that moved
        px = pixel_change_frac(shots[i], shots[i + 1])
        reuse = 1 - shifted
        rel.append((px, shifted))
        print(f"  {lab:30} {px*100:>10.1f}% {shifted*100:>14.1f}% {reuse*100:>9.1f}%")
    print("=" * 74)
    if rel:
        mpx = sum(p for p, _s in rel) / len(rel)
        msh = sum(s for _p, s in rel) / len(rel)
        print(f"  avg pixels changed : {mpx*100:.1f}%   avg vision tokens shifted : {msh*100:.1f}%")
        ratio = msh / mpx if mpx else 0
        verdict = ("LOCAL — unchanged regions are reusable, the 7x is capturable"
                   if msh < mpx * 1.6 else
                   "GLOBAL SPILL — a small change perturbs many tokens; reuse is limited")
        print(f"  token-shift / pixel-change ratio: {ratio:.2f}x  (near 1 = local)  -> {verdict}")
    print("=" * 74)
    hf_cache.commit()


@app.local_entrypoint()
def main() -> None:
    from playwright.sync_api import sync_playwright

    actions = [
        ("type search 'Acme'",  lambda p: p.fill("#search", "Acme")),
        ("open Deals",          lambda p: p.click("#nav-deals")),
        ("select Initech row",  lambda p: p.click("#dealrows tr:nth-child(2)")),
        ("edit amount",         lambda p: p.fill("#amount", "$44,500")),
        ("open stage dropdown", lambda p: p.click("#stage")),
        ("pick Negotiation",    lambda p: p.click("#ddmenu div:nth-child(2)")),
        ("type notes",          lambda p: p.fill("#notes", "Sent revised quote")),
        ("save deal (toast)",   lambda p: p.click("text=Save deal")),
        ("open Contacts",       lambda p: p.click("#nav-contacts")),
        ("back to Deals",       lambda p: p.click("#nav-deals")),
    ]
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={"width": 1280, "height": 800})
        pg.goto(APP.as_uri()); pg.wait_for_timeout(200)
        shots = [pg.screenshot()]; labels = []
        for desc, act in actions:
            act(pg); pg.wait_for_timeout(180)
            shots.append(pg.screenshot()); labels.append(desc)
        b.close()
    shots_b64 = [base64.b64encode(s).decode() for s in shots]
    print(f"captured {len(shots)} frames; measuring vision-embedding stability")
    run.remote(shots_b64, labels)
