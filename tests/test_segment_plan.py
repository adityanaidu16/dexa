"""Pure recompute-planner tests (no model): the causal-dependency logic that
decides what to reuse vs recompute on a context mutation."""

from __future__ import annotations

from dexa.segment import Action, SegmentedContext, Segment, plan_incremental


def _seg(name, toks, role="context"):
    return Segment(name=name, token_ids=tuple(toks), role=role)


def _ctx(*segs):
    return SegmentedContext(list(segs))


def test_cold_start_recomputes_everything():
    new = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4, 5]))
    plan = plan_incremental(None, new)
    assert plan.recompute_tokens == 5
    assert plan.reused_exact_tokens == 0
    assert plan.recompute_ranges() == [(0, 5)]


def test_identical_context_reuses_everything():
    a = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4, 5]))
    b = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4, 5]))
    plan = plan_incremental(a, b)
    assert plan.reused_exact_tokens == 5
    assert plan.recompute_tokens == 0
    assert plan.recompute_ranges() == []
    assert plan.exact


def test_append_reuses_prefix_recomputes_tail():
    a = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4, 5]))
    b = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4, 5]), _seg("turn", [6, 7]))
    plan = plan_incremental(a, b)
    assert plan.reused_exact_tokens == 5          # sys+doc reused
    assert plan.recompute_tokens == 2             # only the appended turn
    assert plan.recompute_ranges() == [(5, 7)]
    s = plan.savings()
    assert s["prefix_reuse_fraction"] == 5 / 7


def test_midcontext_edit_reuses_prefix_recomputes_from_edit():
    # edit the middle doc; everything from it onward must recompute (exact mode).
    a = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4, 5]), _seg("turn", [6, 7]))
    b = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 9, 9, 5]), _seg("turn", [6, 7]))
    plan = plan_incremental(a, b)
    assert plan.reused_exact_tokens == 2          # only sys (before the edit)
    # doc' (4) + turn (2) recomputed; ranges merged and contiguous from edit point
    assert plan.recompute_tokens == 6
    assert plan.recompute_ranges() == [(2, 8)]
    assert plan.exact


def test_edit_at_start_saves_nothing():
    a = _ctx(_seg("sys", [1, 2]), _seg("doc", [3, 4]))
    b = _ctx(_seg("sys", [1, 9]), _seg("doc", [3, 4]))
    plan = plan_incremental(a, b)
    assert plan.reused_exact_tokens == 0
    assert plan.recompute_ranges() == [(0, 4)]


def test_selective_mode_flags_shifted_identical_segment():
    # an upstream doc grows (tool result got longer); the trailing query is content-
    # identical but shifted. exact mode recomputes it; selective flags it for reuse.
    a = _ctx(_seg("sys", [1, 2]), _seg("res", [3, 4]), _seg("q", [7, 8]))
    b = _ctx(_seg("sys", [1, 2]), _seg("res", [3, 4, 5, 6]), _seg("q", [7, 8]))

    exact = plan_incremental(a, b, mode="exact")
    assert exact.exact
    assert exact.recompute_ranges() == [(2, 8)]   # res'(4) + q(2), no reuse of q

    sel = plan_incremental(a, b, mode="selective")
    assert not sel.exact
    q_item = sel.items[-1]
    assert q_item.action == Action.REUSE_SHIFTED
    assert q_item.position_shift == 2             # q moved 2 tokens later
    assert sel.reuse_shifted_tokens == 2
    # only the changed res' is a hard recompute; q is a selective-reuse candidate
    assert sel.recompute_tokens == 4
    assert sel.recompute_ranges() == [(2, 6)]
