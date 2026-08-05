"""A computer-use agent loop on Dexa — shows cumulative impact over a realistic trajectory.

Simulates an agent doing a multi-step task where, like real agents, most steps barely change
the screen (so exact-duplicate re-reads hit the cache, and redundancy stays high). Run the
gateway, run this, then open http://localhost:8080/dashboard to watch the savings accrue.

    python platform/examples/agent_loop.py --steps 20
"""

import argparse
import base64
import io

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dexa-demo")


def frame(step: int, changed: bool) -> str:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1280, 800), "#f4f6f9")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 58], fill="#ffffff")
    d.rectangle([0, 58, 210, 800], fill="#151b2b")
    for i, lbl in enumerate(["Dashboard", "Deals", "Contacts", "Settings"]):
        d.text((24, 90 + i * 34), lbl, fill="#c7d0e0")
    # only a small region changes between "changed" steps (a typed value); mimics real agents
    if changed:
        d.rectangle([300, 300, 700, 340], outline="#2b6cff", width=2)
        d.text((312, 312), f"amount: ${40000 + step * 137}", fill="#131824")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    saved = 0.0
    for s in range(args.steps):
        # every 3rd step the agent re-reads the identical screen (cacheable); others tweak it
        changed = (s % 3 != 0)
        r = client.chat.completions.create(
            model="dexa-cua-vlm",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Next action? CLICK(x,y) / TYPE(text) / DONE."},
                {"type": "image_url", "image_url": {"url": frame(s, changed)}},
            ]}],
            extra_headers={"x-dexa-session": "agent-run-1", "x-dexa-baseline": "gpt-4o"},
        )
        dexa = r.model_dump().get("dexa", {})
        saved += dexa.get("saved_usd", 0.0)
        tag = "cache" if dexa.get("served_from_cache") else f"{dexa.get('screen_redundancy_pct')}% redun"
        print(f"step {s:2d}  {r.choices[0].message.content:14s}  "
              f"${dexa.get('dexa_cost_usd'):.6f} vs ${dexa.get('baseline_cost_usd'):.6f}  [{tag}]")

    print(f"\n{args.steps} steps · saved ${saved:.4f} vs gpt-4o · dashboard: http://localhost:8080/dashboard")


if __name__ == "__main__":
    main()
