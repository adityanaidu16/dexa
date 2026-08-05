"""Session redundancy meter — turns the measured 86%-redundant screen into a live number.

A computer-use agent sends a near-identical screenshot every step. We key frames by a
session id and, per session, measure how much of the screen actually changed since the last
frame (28x28 patches, Qwen's patch size). Two things come out of it:

  1. A visible headroom number — "N% of this frame is unchanged" — the honest quantitative
     signal of how much perception compute is being re-spent on pixels that didn't move.
  2. One real reuse win we can ship today: if a frame is byte-identical to the previous one
     (agents re-read the same screen constantly), we can serve the cached completion and
     skip the model call entirely — a true 100% saving on that step, no accuracy risk.

The bigger prize (reusing compute for the *changed-but-mostly-stable* frame) is gated on the
delta-encoding R&D the evals showed caps at ~2x; this meter measures it but does not claim
it. It reports headroom; it only *acts* on exact duplicates.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

try:
    import numpy as np
    from PIL import Image
    _HAVE_CV = True
except Exception:  # pragma: no cover - degrade to exact-dup-only if imaging libs absent
    _HAVE_CV = False

PATCH = 28
THRESH = 6  # mean abs luma diff for a 28x28 patch to count as "changed"


@dataclass
class Observation:
    first_frame: bool          # no prior frame this session
    exact_duplicate: bool      # byte-identical to previous frame -> cacheable
    changed_frac: float        # fraction of patches that changed (0..1)
    redundant_frac: float      # 1 - changed_frac, the visible headroom


def _patch_grid(img_bytes: bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert("L")
    a = np.asarray(im, dtype=np.int16)
    h = (a.shape[0] // PATCH) * PATCH
    w = (a.shape[1] // PATCH) * PATCH
    if h == 0 or w == 0:
        return None
    return a[:h, :w].reshape(h // PATCH, PATCH, w // PATCH, PATCH).mean(axis=(1, 3))


def _changed_fraction(prev, cur) -> float:
    gh = min(prev.shape[0], cur.shape[0])
    gw = min(prev.shape[1], cur.shape[1])
    d = np.abs(prev[:gh, :gw] - cur[:gh, :gw])
    return float((d > THRESH).mean())


class RedundancyMeter:
    """Per-session frame tracker. One instance per gateway process; keyed by session id."""

    def __init__(self) -> None:
        self._hash: dict[str, str] = {}
        self._grid: dict[str, object] = {}

    def observe(self, session: str, img_bytes: bytes) -> Observation:
        digest = hashlib.sha256(img_bytes).hexdigest()
        prev_hash = self._hash.get(session)
        first = prev_hash is None
        exact = (not first) and digest == prev_hash

        changed = 0.0 if exact else (1.0 if first else None)
        if changed is None and _HAVE_CV:
            cur = _patch_grid(img_bytes)
            prev = self._grid.get(session)
            changed = _changed_fraction(prev, cur) if (prev is not None and cur is not None) else 1.0
            if cur is not None:
                self._grid[session] = cur
        elif changed is None:
            changed = 1.0  # no imaging libs: treat non-identical as fully changed

        self._hash[session] = digest
        if exact is False and _HAVE_CV and session not in self._grid:
            g = _patch_grid(img_bytes)
            if g is not None:
                self._grid[session] = g

        return Observation(
            first_frame=first,
            exact_duplicate=exact,
            changed_frac=changed,
            redundant_frac=max(0.0, 1.0 - changed),
        )
