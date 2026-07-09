"""Engine-side selective recompute across a length-CHANGING edit (Phase 1, Layer C).

Re-phasing alone is never exact for a segment downstream of an edit (it also
attends to the changed content), so the gate here is: (a) the exact prefix is
genuinely reused (bit-identical to prev_kv), and (b) at full recompute the blended
cache reproduces a full re-prefill (behaviorally identical) — proving the assembly
(prefix reuse + re-phased shifted segments + recomputed region) and blend are wired
correctly. The RoPE re-phasing exactness itself is gated in test_rope_rephase.py.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dexa.engine.hf_backend import HFBackend  # noqa: E402
from dexa.segment import Segment, SegmentedContext  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def be():
    return HFBackend(model_name=MODEL, device="cpu", dtype="float32")


def _seg(be, name, text, role="context"):
    return Segment(name=name, token_ids=tuple(be.tokenize(text)), role=role)


def _length_changing_case(be):
    sysg = _seg(be, "sys", "You are an agent.", "system")
    short = Segment("mid", tuple(be.tokenize("Tool result: ok.")), "tool_result")
    long = Segment("mid", tuple(be.tokenize("Tool result: failed with a long detailed traceback here.")), "tool_result")
    tail = _seg(be, "q", "Summarize the outcome now.", "query")
    prev = SegmentedContext([sysg, short, tail])
    new = SegmentedContext([sysg, long, tail])
    assert prev.n_tokens != new.n_tokens          # genuinely length-changing
    return prev, new, sysg


def test_length_changing_full_recompute_matches_full_prefill(be):
    prev, new, sysg = _length_changing_case(be)
    prev_kv = be.prefill(prev.token_ids)
    blended, stats = be.recompute_selective(prev_kv, prev, new, recompute_frac=1.0)
    full = be.prefill(new.token_ids)

    assert stats["length_changed"] is True
    for lb, lf in zip(blended.layers, full.layers):
        assert np.allclose(lb.key, lf.key, atol=1e-5, rtol=1e-4)
        assert np.allclose(lb.value, lf.value, atol=1e-5, rtol=1e-4)
    assert be.generate(blended, [], max_new_tokens=8) == be.generate(full, [], max_new_tokens=8)


def test_length_changing_reuses_exact_prefix(be):
    prev, new, sysg = _length_changing_case(be)
    prev_kv = be.prefill(prev.token_ids)
    blended, _ = be.recompute_selective(prev_kv, prev, new, recompute_frac=0.0)
    p = sysg.n_tokens
    # the system prefix is copied verbatim from prev_kv (genuine reuse, not recompute)
    for lb, lp in zip(blended.layers, prev_kv.layers):
        assert np.array_equal(lb.key[:, :p], lp.key[:, :p])
        assert np.array_equal(lb.value[:, :p], lp.value[:, :p])
