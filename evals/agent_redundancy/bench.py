"""How much does a computer-use agent's screen actually change per action?

Drives a realistic CRM web app with Playwright (a stand-in agent doing a real task),
captures the screenshot after every action, and measures the fraction of 28x28 patches
(Qwen2.5-VL's patch size) that change between consecutive frames. That fraction is the
share of visual tokens that would need RE-ENCODING each step; 1 - it is the redundancy a
delta/KV-reuse perception engine could exploit. Big redundancy => the "encode only what
changed" thesis is a large, real prize on the workload frontier labs serve statelessly.

    python evals/agent_redundancy/bench.py
"""

import io
import pathlib

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
APP = pathlib.Path(__file__).parent / "app.html"
PATCH = 28
THRESH = 6  # mean abs luma diff for a patch to count as "changed"


def changed_fraction(a: bytes, b: bytes) -> float:
    A = np.asarray(Image.open(io.BytesIO(a)).convert("L"), dtype=np.int16)
    B = np.asarray(Image.open(io.BytesIO(b)).convert("L"), dtype=np.int16)
    h, w = (min(A.shape[0], B.shape[0]) // PATCH) * PATCH, (min(A.shape[1], B.shape[1]) // PATCH) * PATCH
    A, B = A[:h, :w], B[:h, :w]
    d = np.abs(A - B).reshape(h // PATCH, PATCH, w // PATCH, PATCH).mean(axis=(1, 3))
    return float((d > THRESH).mean())


def run():
    # a realistic agent task: find a deal, edit it, change stage, save, then navigate
    actions = [
        ("type search 'Acme'",      lambda p: p.fill("#search", "Acme")),
        ("open Deals",              lambda p: p.click("#nav-deals")),
        ("select Initech row",      lambda p: p.click("#dealrows tr:nth-child(2)")),
        ("edit amount",             lambda p: p.fill("#amount", "$44,500")),
        ("focus close date",        lambda p: p.fill("#cdate", "2026-04-15")),
        ("open stage dropdown",     lambda p: p.click("#stage")),
        ("pick Negotiation",        lambda p: p.click("#ddmenu div:nth-child(2)")),
        ("type notes",             lambda p: p.fill("#notes", "Sent revised quote")),
        ("save deal (toast)",       lambda p: p.click("text=Save deal")),
        ("select Stark row",        lambda p: p.click("#dealrows tr:nth-child(4)")),
        ("edit amount again",       lambda p: p.fill("#amount", "$121,000")),
        ("open Contacts",           lambda p: p.click("#nav-contacts")),
        ("open Dashboard",          lambda p: p.click("#nav-dash")),
        ("open Settings",           lambda p: p.click("#nav-settings")),
        ("back to Deals",           lambda p: p.click("#nav-deals")),
    ]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(APP.as_uri())
        page.wait_for_timeout(200)
        shots = [page.screenshot()]
        labels = []
        for desc, act in actions:
            act(page)
            page.wait_for_timeout(180)
            shots.append(page.screenshot())
            labels.append(desc)
        browser.close()

    fracs = [changed_fraction(shots[i], shots[i + 1]) for i in range(len(shots) - 1)]
    avg = sum(fracs) / len(fracs)

    print("\n" + "=" * 66)
    print("AGENT SCREEN REDUNDANCY — Zenith CRM, 15-step task (Chromium)")
    print("=" * 66)
    print(f"  {'step':32} {'% patches changed':>18}")
    for lab, f in zip(labels, fracs):
        bar = "█" * max(1, round(f * 30))
        print(f"  {lab:32} {f*100:>7.1f}%  {bar}")
    print("=" * 66)
    red = 1 - avg
    print(f"  avg patches changed per action : {avg*100:.1f}%")
    print(f"  avg redundancy (reusable)      : {red*100:.1f}%")
    print(f"  implied encode-only-changed headroom: {1/avg:.1f}x fewer visual tokens/step")
    print("=" * 66)
    print(f"  Over a {len(actions)}-step trajectory: naive re-encodes {len(actions)+1} full frames;")
    print(f"  a delta engine encodes 1 full frame + {len(actions)} deltas of ~{avg*100:.0f}% each.")
    print("=" * 66)


if __name__ == "__main__":
    run()
