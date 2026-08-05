"""Cost accounting — the quantitative-impact core.

For a computer-use agent, the request is a screenshot + a bit of text, and the cost is
dominated by how many *visual tokens* the provider bills for that screenshot. Different
providers count the same image wildly differently, so the honest way to show impact is to
compute, for the exact image the caller sent, what it costs on Dexa vs what it would have
cost on the frontier model they're switching from — using each provider's real image-token
accounting, not a hand-wave.

Everything here is deterministic and unit-tested. The dollar rates are published list
prices (see PRICES) and are the one thing a customer can audit and override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---- published list prices, USD per 1M tokens (input, output) ---------------------------
# These are the auditable knobs. Update PRICES; everything downstream is derived.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o":         (2.50, 10.00),
    "gpt-4o-mini":    (0.15,  0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    # Dexa serves an open VLM (Qwen2.5-VL-7B) — open-weight 7B economics, one blended rate.
    "dexa-cua-vlm":   (0.20,  0.20),
}

# which token-accounting scheme each model uses for images
_SCHEME = {
    "gpt-4o": "openai", "gpt-4o-mini": "openai_mini",
    "claude-3-5-sonnet": "anthropic", "dexa-cua-vlm": "qwen",
}


# ---- per-provider image -> token accounting ---------------------------------------------
def qwen_vision_tokens(w: int, h: int, max_pixels: int = 1_048_576,
                       min_pixels: int = 3136, patch: int = 28, merge: int = 2) -> int:
    """Qwen2.5-VL: smart-resize to a multiple of patch*merge under the pixel budget, then
    every patch*merge x patch*merge block becomes one token fed to the LLM."""
    factor = patch * merge
    if w <= 0 or h <= 0:
        return 0
    # smart_resize: keep aspect, round each side to a multiple of `factor`, clamp total px
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        s = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / s / factor) * factor)
        wb = max(factor, math.floor(w / s / factor) * factor)
    elif hb * wb < min_pixels:
        s = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * s / factor) * factor
        wb = math.ceil(w * s / factor) * factor
    return (hb // factor) * (wb // factor)


def _openai_tiles(w: int, h: int) -> int:
    """OpenAI high-detail tiling: fit within 2048 long / 768 short, then count 512px tiles."""
    if max(w, h) > 2048:
        s = 2048 / max(w, h)
        w, h = round(w * s), round(h * s)
    if min(w, h) > 768:
        s = 768 / min(w, h)
        w, h = round(w * s), round(h * s)
    return math.ceil(w / 512) * math.ceil(h / 512)


def openai_image_tokens(w: int, h: int, mini: bool = False) -> int:
    tiles = _openai_tiles(w, h)
    # documented multipliers: 4o = 85 base + 170/tile; 4o-mini = 2833 base + 5667/tile
    return (2833 + 5667 * tiles) if mini else (85 + 170 * tiles)


def anthropic_image_tokens(w: int, h: int) -> int:
    """Claude bills images at roughly (w*h)/750 tokens (capped by its own resize to ~1.15MP)."""
    if w * h > 1_150_000:
        s = math.sqrt(1_150_000 / (w * h))
        w, h = round(w * s), round(h * s)
    return math.ceil((w * h) / 750)


def image_tokens(model: str, w: int, h: int) -> int:
    scheme = _SCHEME[model]
    if scheme == "qwen":
        return qwen_vision_tokens(w, h)
    if scheme == "openai":
        return openai_image_tokens(w, h, mini=False)
    if scheme == "openai_mini":
        return openai_image_tokens(w, h, mini=True)
    if scheme == "anthropic":
        return anthropic_image_tokens(w, h)
    raise KeyError(model)


# ---- request-level cost ------------------------------------------------------------------
@dataclass
class Cost:
    model: str
    image_tokens: int
    text_in_tokens: int
    out_tokens: int
    usd: float

    @property
    def in_tokens(self) -> int:
        return self.image_tokens + self.text_in_tokens


def request_cost(model: str, images: list[tuple[int, int]], text_in_tokens: int,
                 out_tokens: int) -> Cost:
    img_tok = sum(image_tokens(model, w, h) for (w, h) in images)
    pin, pout = PRICES[model]
    usd = (img_tok + text_in_tokens) / 1e6 * pin + out_tokens / 1e6 * pout
    return Cost(model, img_tok, text_in_tokens, out_tokens, usd)


@dataclass
class Savings:
    ours: Cost
    baseline: Cost
    saved_usd: float
    saved_pct: float
    x_cheaper: float

    def as_dict(self) -> dict:
        # x_cheaper is infinite when our cost is ~0 (e.g. served from cache); JSON has no
        # inf, so surface it as None and let the client read "free" from saved_pct == 100.
        x = None if math.isinf(self.x_cheaper) else round(self.x_cheaper, 2)
        return {
            "dexa_cost_usd": round(self.ours.usd, 8),
            "baseline_model": self.baseline.model,
            "baseline_cost_usd": round(self.baseline.usd, 8),
            "saved_usd": round(self.saved_usd, 8),
            "saved_pct": round(self.saved_pct, 2),
            "x_cheaper": x,
            "dexa_image_tokens": self.ours.image_tokens,
            "baseline_image_tokens": self.baseline.image_tokens,
        }


def compare(images: list[tuple[int, int]], text_in_tokens: int, out_tokens: int,
            baseline_model: str = "gpt-4o", our_model: str = "dexa-cua-vlm") -> Savings:
    ours = request_cost(our_model, images, text_in_tokens, out_tokens)
    base = request_cost(baseline_model, images, text_in_tokens, out_tokens)
    saved = base.usd - ours.usd
    pct = (saved / base.usd * 100) if base.usd > 0 else 0.0
    x = (base.usd / ours.usd) if ours.usd > 0 else float("inf")
    return Savings(ours, base, saved, pct, x)
