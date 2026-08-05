"""Threshold sweep on the delta-perception embeddings — is the vision-token 'spill' soft or hard?

The viability run counted a token 'shifted' at a single cutoff (cosine < 0.98) and found
~53% of tokens move for a ~16% pixel change (3.4x spill). But a token perturbed to cosine
0.995 is, for practical purposes, the SAME token — reusing it is almost certainly fine. So
the single 0.98 cutoff conflates 'genuinely re-encoded' with 'nudged but reusable'.

This re-scores the SAME frames (no new assumptions, just a finer ruler) across a ladder of
tolerances and dumps the full cosine distribution of the tokens that moved. The question it
answers: of the 53% spill, how much sits just below 0.98 (soft — reuse with tolerance) vs
far below (hard — must re-encode)? That sets the realistic upper bound on capturable reuse
BEFORE the expensive grounding-accuracy test.

It also splits small-action frames (typing/editing/clicking — the common agent step) from
full-page-nav keyframes (>30% pixels), since keyframes get re-encoded regardless and
shouldn't count against the reuse ceiling.

    modal run evals/agent_redundancy/modal_delta_tolerance.py
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
app = modal.App("dexa-delta-tolerance")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

# tolerance ladder: a token with cosine >= T is deemed reusable at tolerance T
TOLS = [0.98, 0.99, 0.995, 0.999, 0.9999]
KEYFRAME_PX = 0.30  # a frame whose pixels changed > this is a full-page nav -> re-encode anyway


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

    # per-frame per-token cosine + pixel change; keep only comparable (same-grid) pairs
    frames = []  # (label, px_frac, cos_tensor[n_tokens])
    for i, lab in enumerate(labels):
        e0, g0 = embs[i]; e1, g1 = embs[i + 1]
        if g0 != g1 or e0.shape != e1.shape:
            continue
        cos = torch.nn.functional.cosine_similarity(e0, e1, dim=-1).cpu().numpy()
        frames.append((lab, pixel_change_frac(shots[i], shots[i + 1]), cos))

    small = [f for f in frames if f[1] <= KEYFRAME_PX]
    keyf = [f for f in frames if f[1] > KEYFRAME_PX]

    def reusable_at(cos_list, tol):
        # fraction of tokens with cosine >= tol, pooled across the given frames
        allc = np.concatenate(cos_list) if cos_list else np.array([])
        return float((allc >= tol).mean()) if allc.size else 0.0

    print("\n" + "=" * 78)
    print("DELTA TOLERANCE SWEEP — is the vision-token spill soft (reusable) or hard?")
    print("=" * 78)

    # 1) reusable fraction & implied per-step multiple vs tolerance, small-action frames
    small_cos = [c for _l, _p, c in small]
    print(f"\n  SMALL-ACTION frames only (n={len(small)}; typing/editing/clicking — the common step)")
    print(f"  {'tolerance (cos>=)':>20} {'reusable tokens':>16} {'re-encode':>11} {'per-step mult':>14}")
    for t in TOLS:
        r = reusable_at(small_cos, t)
        reenc = 1 - r
        mult = (1 / reenc) if reenc > 1e-6 else float("inf")
        print(f"  {t:>20.4f} {r*100:>15.1f}% {reenc*100:>10.1f}% {mult:>13.2f}x")

    # 2) full cosine distribution of the tokens that 'moved' at the original 0.98 cutoff
    #    (small-action frames) — where does the spill mass actually sit?
    allc_small = np.concatenate(small_cos) if small_cos else np.array([])
    moved = allc_small[allc_small < 0.98]
    print(f"\n  Distribution of the tokens that MOVED (cosine < 0.98) on small actions")
    print(f"  {'cosine band':>16} {'share of moved':>16}   (soft = high cosine, hard = low)")
    bands = [(0.95, 0.98, "0.95–0.98  soft"), (0.90, 0.95, "0.90–0.95"),
             (0.80, 0.90, "0.80–0.90"), (0.50, 0.80, "0.50–0.80"),
             (-1.0, 0.50, "<0.50  hard")]
    for lo, hi, lab in bands:
        share = float(((moved >= lo) & (moved < hi)).mean()) if moved.size else 0.0
        bar = "#" * max(0, round(share * 40))
        print(f"  {lab:>16} {share*100:>15.1f}%   {bar}")
    if moved.size:
        print(f"  median cosine of moved tokens: {float(np.median(moved)):.4f}   "
              f"(closer to 0.98 = softer, more reusable)")

    # 3) keyframe frames, for completeness
    if keyf:
        kf_cos = [c for _l, _p, c in keyf]
        print(f"\n  KEYFRAME frames (>{KEYFRAME_PX*100:.0f}% pixels — full-page nav, re-encoded regardless; n={len(keyf)})")
        for t in (0.98, 0.999):
            print(f"    reusable at cos>={t}: {reusable_at(kf_cos, t)*100:.1f}%")

    print("\n" + "=" * 78)
    r98 = reusable_at(small_cos, 0.98)
    r999 = reusable_at(small_cos, 0.999)
    m98 = 1 / (1 - r98) if (1 - r98) > 1e-6 else float("inf")
    m999 = 1 / (1 - r999) if (1 - r999) > 1e-6 else float("inf")
    print(f"  Small-action per-step multiple: {m98:.2f}x at strict 0.98  ->  {m999:.2f}x at loose 0.999")
    if m999 >= m98 * 1.5:
        print("  VERDICT: much of the spill is SOFT — tolerant reuse meaningfully raises the ceiling.")
        print("           The 5x case is alive; the grounding-accuracy test is worth running.")
    else:
        print("  VERDICT: the spill is mostly HARD — loosening tolerance barely helps.")
        print("           The ceiling really is ~2x; don't fund the accuracy sprint on this alone.")
    print("  (Ceiling only — true reuse still bounded by whether accuracy holds on reused tokens.)")
    print("=" * 78)
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
    print(f"captured {len(shots)} frames; sweeping reuse tolerance")
    run.remote(shots_b64, labels)
