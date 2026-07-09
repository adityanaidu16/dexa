"""Exact incremental recompute on the real HF backend (Phase 1, Layer B).

The gate: reusing the unchanged segment prefix and recomputing only from the edit
onward must (a) genuinely reuse the prefix KV — bit-identical to prev_kv — and
(b) produce a result numerically equivalent to a full re-prefill of the mutated
context, with an identical greedy continuation. Exercised with real causal
attention + RoPE + GQA (tiny-random Llama, CPU); if prefix reuse were subtly wrong
(positions, RoPE phase, attention over the reused region) the continuation would
diverge. See :func:`_assert_equivalent`.
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


def _assert_equivalent(be, kv, full, prev_kv, prefix):
    """The reuse claim + the correctness claim.

    * Reuse: ``kv``'s first ``prefix`` tokens are **bit-identical to prev_kv** —
      proving they were genuinely reused, not silently recomputed. (Against a fresh
      full prefill they are only fp-close: SDPA picks different kernel tiling at
      different sequence lengths, ~1e-8, even for causally-masked prefix rows.)
    * Correctness: ``kv`` is numerically equivalent to a full re-prefill (a
      KV-cached forward differs from one full forward only by fp reduction order),
      and the greedy continuation is identical (the behavioral guarantee)."""
    assert kv.seq_len == full.seq_len and kv.token_ids == full.token_ids
    assert np.array_equal(kv.positions, full.positions)
    for lk, lp in zip(kv.layers, prev_kv.layers):
        assert np.array_equal(lk.key[:, :prefix], lp.key[:, :prefix])   # reused, not recomputed
        assert np.array_equal(lk.value[:, :prefix], lp.value[:, :prefix])
    for lk, lf in zip(kv.layers, full.layers):
        assert np.allclose(lk.key, lf.key, atol=1e-5, rtol=1e-4)        # == full prefill
        assert np.allclose(lk.value, lf.value, atol=1e-5, rtol=1e-4)
    assert be.generate(kv, [], max_new_tokens=8) == be.generate(full, [], max_new_tokens=8)


def test_append_is_bit_identical_and_reuses_prefix(be):
    prev = SegmentedContext([_seg(be, "sys", "You are a helpful assistant."),
                             _seg(be, "doc", "The capital of France is Paris.")])
    new = SegmentedContext(prev.segments + [_seg(be, "turn", "What is the capital?", "query")])

    prev_kv = be.prefill(prev.token_ids)
    kv, stats = be.recompute_incremental(prev_kv, prev, new)
    full = be.prefill(new.token_ids)

    _assert_equivalent(be, kv, full, prev_kv, stats["reused_tokens"])
    assert stats["reused_tokens"] == prev.n_tokens
    assert stats["recomputed_tokens"] == new.segments[-1].n_tokens


def test_midcontext_edit_is_equivalent(be):
    a_doc = _seg(be, "doc", "The tool returned status code 200.")
    b_doc = _seg(be, "doc", "The tool returned an error: connection refused after retry.")
    sysseg = _seg(be, "sys", "You are a coding agent operating on a repository.")
    tail = _seg(be, "turn", "Summarize what happened.", "query")

    prev = SegmentedContext([sysseg, a_doc, tail])
    new = SegmentedContext([sysseg, b_doc, tail])

    prev_kv = be.prefill(prev.token_ids)
    kv, stats = be.recompute_incremental(prev_kv, prev, new)
    full = be.prefill(new.token_ids)

    _assert_equivalent(be, kv, full, prev_kv, stats["reused_tokens"])   # equivalence despite mid-context edit
    assert stats["reused_tokens"] == sysseg.n_tokens   # only the system prefix reused
    assert stats["recomputed_tokens"] == b_doc.n_tokens + tail.n_tokens


def test_identical_context_reuses_all_no_recompute(be):
    ctx = SegmentedContext([_seg(be, "sys", "System."), _seg(be, "doc", "Body text here.")])
    prev_kv = be.prefill(ctx.token_ids)
    kv, stats = be.recompute_incremental(prev_kv, ctx, ctx)
    assert stats["recomputed_tokens"] == 0
    _assert_equivalent(be, kv, be.prefill(ctx.token_ids), prev_kv, stats["reused_tokens"])
