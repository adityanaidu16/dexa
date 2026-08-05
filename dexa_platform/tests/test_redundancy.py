"""Redundancy meter — exact-dup detection and headroom measurement."""

import io

import pytest

from dexa_platform.gateway.redundancy import RedundancyMeter

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def png(draw_fn=None, size=(1280, 800)) -> bytes:
    im = Image.new("RGB", size, "#f4f6f9")
    if draw_fn:
        draw_fn(ImageDraw.Draw(im))
    b = io.BytesIO(); im.save(b, format="PNG")
    return b.getvalue()


def test_first_frame_has_no_redundancy():
    m = RedundancyMeter()
    o = m.observe("s", png())
    assert o.first_frame and not o.exact_duplicate
    assert o.redundant_frac == 0.0


def test_identical_frame_is_exact_duplicate():
    m = RedundancyMeter()
    frame = png(lambda d: d.rectangle([100, 100, 300, 200], fill="#2b6cff"))
    m.observe("s", frame)
    o = m.observe("s", frame)
    assert o.exact_duplicate
    assert o.redundant_frac == 1.0


def test_small_change_is_mostly_redundant():
    m = RedundancyMeter()
    m.observe("s", png())
    o = m.observe("s", png(lambda d: d.rectangle([300, 300, 360, 340], fill="#111")))
    assert not o.exact_duplicate
    assert o.redundant_frac > 0.9      # a tiny box changes <10% of patches


def test_full_repaint_is_not_redundant():
    m = RedundancyMeter()
    m.observe("s", png())
    o = m.observe("s", png(lambda d: d.rectangle([0, 0, 1280, 800], fill="#151b2b")))
    assert o.redundant_frac < 0.2


def test_sessions_are_isolated():
    m = RedundancyMeter()
    f = png(lambda d: d.rectangle([10, 10, 50, 50], fill="#000"))
    m.observe("a", f)
    o = m.observe("b", f)            # same bytes, different session -> first frame, not a dup
    assert o.first_frame and not o.exact_duplicate
