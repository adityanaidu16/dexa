"""Demo: calling the doc-VLM endpoint is identical to calling OpenAI — one line changed.

    python serve/demo_client.py

The ONLY difference from an OpenAI vision call is base_url. Same client, same
chat/completions API, same image_url content — served ~40x cheaper by our optimized
Qwen2.5-VL. First call cold-starts the GPU (~1-2 min); after that it's warm.
"""

import base64
import io

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "https://adityanaidu16344--dexa-doc-vlm-serve.modal.run/v1"


def synthetic_invoice() -> str:
    img = Image.new("RGB", (1000, 520), "white")
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.load_default(size=44)
        med = ImageFont.load_default(size=34)
    except TypeError:
        big = med = ImageFont.load_default()
    d.text((50, 40), "ACME Corp", fill="black", font=big)
    d.text((50, 110), "Invoice #4471   Date: 2026-02-14", fill="black", font=med)
    d.text((50, 220), "Subtotal ................. $1,180.00", fill="black", font=med)
    d.text((50, 270), "Tax (4.62%) ............... $54.56", fill="black", font=med)
    d.text((50, 340), "TOTAL DUE ................. $1,234.56", fill="black", font=big)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")  # <-- the only change
    for q in ["What is the total due? Answer with just the amount.",
              "What is the invoice number?",
              "What is the tax rate?"]:
        r = client.chat.completions.create(
            model="doc-vlm", max_tokens=32, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {"url": synthetic_invoice()}}]}])
        print(f"Q: {q}\nA: {r.choices[0].message.content.strip()}\n")


if __name__ == "__main__":
    main()
