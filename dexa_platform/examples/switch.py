"""The entire switching cost, in one file.

This is the code a computer-use agent team already has. To move onto Dexa they change two
lines — base_url and api_key — and nothing else. Same OpenAI SDK, same message shape, same
image_url content. The savings come back in the response under `.dexa`.

    pip install openai
    python platform/examples/switch.py
"""

import base64
import io

from openai import OpenAI

# --- before ---------------------------------------------------------------
# client = OpenAI(api_key="sk-...")                       # OpenAI
# MODEL = "gpt-4o"

# --- after (the whole diff) ----------------------------------------------
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dexa-demo")
MODEL = "dexa-cua-vlm"
# -------------------------------------------------------------------------


def screenshot_data_uri() -> str:
    """Stand-in for your agent's real screenshot grab."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1280, 800), "#f4f6f9")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1280, 58], fill="#ffffff")
    d.rectangle([40, 120, 1240, 170], outline="#2b6cff", width=2)
    d.text((60, 136), "Deals  >  Initech  >  Edit amount", fill="#131824")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "You control a browser. What single action next? Reply CLICK(x,y) or TYPE(text)."},
        {"type": "image_url", "image_url": {"url": screenshot_data_uri()}},
    ]}],
    extra_headers={"x-dexa-session": "demo-switch", "x-dexa-baseline": "gpt-4o"},
)

print("action:", resp.choices[0].message.content)
# the impact, right in the response object:
dexa = resp.model_dump().get("dexa", {})
print(f"this call: ${dexa.get('dexa_cost_usd'):.6f}  "
      f"(gpt-4o would be ${dexa.get('baseline_cost_usd'):.6f} "
      f"-> {dexa.get('x_cheaper')}x cheaper, {dexa.get('saved_pct')}% off)")
print(f"screen redundancy this step: {dexa.get('screen_redundancy_pct')}%")
