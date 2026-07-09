"""SegmentedSession: mutation-equivalence, branching, rollback, diff, persistence.

Uses the real HF backend (tiny-random Llama, CPU) so mutation KV really goes
through incremental recompute; equivalence to a full re-prefill is checked
behaviorally (greedy continuation), the honest guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dexa.engine.hf_backend import HFBackend  # noqa: E402
from dexa.segment import Segment, SegmentedContext, SegmentedSession  # noqa: E402
from dexa.session.store import SessionStore  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def be():
    return HFBackend(model_name=MODEL, device="cpu", dtype="float32")


def _seg(be, name, text, role="context"):
    return Segment(name=name, token_ids=tuple(be.tokenize(text)), role=role)


def _same_behavior(be, kv, segments):
    """Session KV must behave like a fresh full prefill of the same segments."""
    full = be.prefill(SegmentedContext(list(segments)).token_ids)
    return be.generate(kv, [], max_new_tokens=8) == be.generate(full, [], max_new_tokens=8)


def test_mutations_stay_equivalent_and_save_tokens(be):
    sess = SegmentedSession(be, [_seg(be, "sys", "You are an agent."),
                                 _seg(be, "doc", "Repo file contents here.")])
    sess.append(_seg(be, "turn1", "User: run the tests.", "query"))
    sess.append(_seg(be, "res1", "Tests passed: 42 ok.", "tool_result"))
    sess.edit("res1", _seg(be, "res1", "Tests failed: 3 errors.", "tool_result"))

    assert _same_behavior(be, sess.kv, sess.segments)
    # incremental reprocessed strictly fewer tokens than repeated full re-prefill.
    assert sess.stats["tokens_recomputed"] < sess.stats["tokens_full_reprefill"]


def test_branch_is_isolated_and_shares_no_state(be):
    parent = SegmentedSession(be, [_seg(be, "sys", "Shared system prompt.")])
    parent.append(_seg(be, "ctx", "Shared context body."))
    child = parent.branch()

    parent.append(_seg(be, "p", "Parent-only turn.", "query"))
    child.append(_seg(be, "c", "Child-only different turn.", "query"))

    assert [s.name for s in parent.segments] == ["sys", "ctx", "p"]
    assert [s.name for s in child.segments] == ["sys", "ctx", "c"]
    assert _same_behavior(be, parent.kv, parent.segments)
    assert _same_behavior(be, child.kv, child.segments)
    # mutating the child did not alter the parent's KV (no aliasing).
    assert parent.kv.seq_len != child.kv.seq_len or not np.array_equal(
        parent.kv.layers[0].key, child.kv.layers[0].key)


def test_commit_rollback_restores(be):
    sess = SegmentedSession(be, [_seg(be, "sys", "System.")])
    sess.append(_seg(be, "good", "A good turn.", "query"))
    sess.commit("v1")
    good_names = [s.name for s in sess.segments]

    sess.append(_seg(be, "bad", "A speculative bad turn.", "query"))
    assert [s.name for s in sess.segments] != good_names

    sess.rollback("v1")
    assert [s.name for s in sess.segments] == good_names
    assert _same_behavior(be, sess.kv, sess.segments)
    assert "v1" in sess.versions()


def test_diff_reports_common_prefix_and_tails(be):
    a = SegmentedSession(be, [_seg(be, "sys", "Sys."), _seg(be, "x", "X body.")])
    b = a.branch()
    a.append(_seg(be, "a1", "alpha", "query"))
    b.append(_seg(be, "b1", "beta", "query"))
    d = a.diff(b)
    assert d["common_prefix_segments"] == 2
    assert d["only_self"] == ["a1"] and d["only_other"] == ["b1"]


def test_persist_and_reattach(be, tmp_path):
    segs = [_seg(be, "sys", "System prompt."), _seg(be, "doc", "Document body here.")]
    sess = SegmentedSession(be, segs)
    sess.append(_seg(be, "turn", "A question.", "query"))
    store = SessionStore(tmp_path / "sess", format="blob")
    sess.save(store, "s1")

    reattached = SegmentedSession.load(be, store, "s1", sess.segments)
    assert _same_behavior(be, reattached.kv, reattached.segments)
