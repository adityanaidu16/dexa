"""Engine-agnostic session lifecycle — delta prefill + persist/resume.

FakeBackend is deterministic and supports extend/generate_and_extend, so this
exercises the full lifecycle (multi-turn growth, the prefill-savings accounting,
and restore-after-restart) without a GPU.
"""

from __future__ import annotations

from dexa.engine.fake import FakeBackend
from dexa.serving import SessionManager
from dexa.session.store import SessionStore


def test_multi_turn_delta_prefill(tmp_path):
    mgr = SessionManager(FakeBackend(), store=SessionStore(tmp_path / "s"))
    sid = "a"
    _, i1 = mgr.turn(sid, "hello there friend", system="be terse", max_new_tokens=4)
    _, i2 = mgr.turn(sid, "what did I say", max_new_tokens=4)
    _, i3 = mgr.turn(sid, "and again please", max_new_tokens=4)

    # context grows turn over turn
    assert i1.context_tokens < i2.context_tokens < i3.context_tokens
    assert i1.turns == 1 and i3.turns == 3
    # later turns prefill only the small delta, not the whole history
    assert i2.prefill_delta_tokens < i2.stateless_would_prefill
    assert i2.prefill_savings > 0.0
    assert i3.prefill_savings > i2.prefill_savings  # savings grow as context grows


def test_resume_after_restart(tmp_path):
    store_dir = tmp_path / "s"
    mgr = SessionManager(FakeBackend(), store=SessionStore(store_dir))
    mgr.turn("sess", "first turn here", system="sys", max_new_tokens=4)
    ctx_before = mgr.turn("sess", "second turn", max_new_tokens=4)[1].context_tokens

    # fresh manager (new process), same store -> must restore the session
    mgr2 = SessionManager(FakeBackend(), store=SessionStore(store_dir))
    assert mgr2.exists("sess")
    _, info = mgr2.turn("sess", "third turn after restart", max_new_tokens=4)
    assert info.resumed is True
    assert info.context_tokens > ctx_before          # continued, didn't restart
    assert info.turns == 3
