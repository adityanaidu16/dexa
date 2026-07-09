"""On-device KV session (TorchKVSession): correctness parity with full prefill.

The wall-time win of keeping KV on-device only counts if it stays exact. These
check — on the real tiny Llama, CPU — that append/edit on the resident cache
produce behavior identical to a full re-prefill, and that the on-device decode
matches the backend's numpy decode. Wall-time itself is measured on GPU
(benchmarks/ondevice_incremental_bench.py); parity is the precondition.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dexa.engine.hf_backend import HFBackend  # noqa: E402
from dexa.engine.torch_session import TorchKVSession  # noqa: E402
from dexa.segment import Segment, SegmentedContext  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def be():
    return HFBackend(model_name=MODEL, device="cpu", dtype="float32")


def _seg(be, name, text, role="context"):
    return Segment(name=name, token_ids=tuple(be.tokenize(text)), role=role)


def test_append_matches_full_prefill(be):
    segs = [_seg(be, "sys", "You are an agent."),
            _seg(be, "doc", "Some repository context."),
            _seg(be, "turn", "Run the tests please.", "query")]
    sess = TorchKVSession(be)
    for s in segs:
        sess.append(list(s.token_ids))

    full = be.prefill(SegmentedContext(segs).token_ids)
    # on-device decode == numpy full-prefill decode
    assert sess.greedy(8) == be.generate(full, [], max_new_tokens=8)
    # materialized session KV decodes the same too
    assert be.generate(sess.to_kvcache(), [], max_new_tokens=8) == be.generate(full, [], max_new_tokens=8)


def test_midcontext_edit_on_device_matches_full(be):
    sysg = _seg(be, "sys", "You are a coding agent.")
    old = _seg(be, "doc", "Tool returned status 200 ok.")
    new = _seg(be, "doc", "Tool returned an error and a long traceback.")
    tail = _seg(be, "q", "Summarize what happened.", "query")

    prev = SegmentedContext([sysg, old, tail])
    new_ctx = SegmentedContext([sysg, new, tail])

    sess = TorchKVSession(be, prev.token_ids)
    stats = sess.apply(prev, new_ctx)
    assert stats["reused_tokens"] == sysg.n_tokens          # only the prefix reused
    assert sess.token_ids == new_ctx.token_ids

    full = be.prefill(new_ctx.token_ids)
    assert sess.greedy(8) == be.generate(full, [], max_new_tokens=8)


def test_truncate_shrinks_resident_cache(be):
    sess = TorchKVSession(be, be.tokenize("one two three four five six seven eight"))
    n0 = sess.n
    sess.truncate(3)
    assert sess.n == 3
    assert sess.cache.layers[0].keys.shape[2] == 3          # seq dim really sliced
    assert n0 > 3
