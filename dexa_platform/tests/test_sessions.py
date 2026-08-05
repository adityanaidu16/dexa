"""Tiering policy, session store, and the turn flow (mock backend, no GPU)."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["DEXA_SESSION_MOCK"] = "1"

from dexa_platform.sessions import tiering            # noqa: E402
from dexa_platform.sessions.service import app        # noqa: E402
from dexa_platform.sessions.store import SessionStore, est_tokens  # noqa: E402


# ---- tiering policy: boundaries come from the measured break-evens ----------------------
def test_breakevens_grow_with_context():
    be_small = tiering.breakevens(4096)
    be_big = tiering.breakevens(65536)
    # NVMe break-even is much longer for bigger context (costlier prefill to avoid)
    assert be_big.nvme_s > be_small.nvme_s
    # ordering warm < ram < nvme always
    for be in (be_small, be_big):
        assert be.warm_s < be.ram_s < be.nvme_s


def test_policy_picks_tier_by_idle():
    p = tiering.TieringPolicy()
    assert p.decide(idle_s=5, tokens=65536).tier == "warm"        # just idled
    assert p.decide(idle_s=300, tokens=65536).tier == "ram"       # 5 min
    assert p.decide(idle_s=3600, tokens=65536).tier == "nvme"     # 1 hr -> NVMe still pays at 64k
    assert p.decide(idle_s=10 * 3600, tokens=65536).tier == "drop"  # 10 hr -> re-prefill


def test_savings_positive_when_warm():
    s = tiering.estimated_savings(65536, warm=True)              # restore instead of prefill
    assert s["saved_ms"] > 0 and s["speedup"] > 5
    cold = tiering.estimated_savings(65536, warm=False)          # first touch is the prefill
    assert cold["saved_ms"] == 0


# ---- store ------------------------------------------------------------------------------
def test_est_tokens_counts_text():
    n = est_tokens([{"role": "user", "content": "x" * 4000}])
    assert 900 < n < 1100


def test_store_turn_accumulates():
    st = SessionStore()
    s = st.create("m", [{"role": "system", "content": "hi"}])
    st.record_turn(s, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                   tier="ram", saved={"saved_usd": 0.01, "saved_ms": 100})
    assert s.turns == 1 and s.tier == "ram" and s.saved_usd == 0.01
    assert len(s.messages) == 3


# ---- end-to-end turn flow (mock backend) ------------------------------------------------
def test_first_turn_cold_second_turn_warm():
    c = TestClient(app)
    sid = c.post("/v1/sessions", json={"context": "C" * 60000}).json()["session"]["id"]

    t1 = c.post(f"/v1/sessions/{sid}/turn", json={"content": "what is here?"}).json()
    assert t1["turn"]["warm"] is False                # first touch -> cold prefill
    assert t1["savings_vs_stateless"]["saved_ms"] == 0

    t2 = c.post(f"/v1/sessions/{sid}/turn", json={"content": "and now?"}).json()
    assert t2["turn"]["warm"] is True                 # KV restored -> warm
    assert t2["turn"]["latency_ms"] < t1["turn"]["latency_ms"]
    assert t2["savings_vs_stateless"]["saved_ms"] > 0
    assert t2["savings_vs_stateless"]["speedup"] > 3


def test_delete_frees_session():
    c = TestClient(app)
    sid = c.post("/v1/sessions", json={}).json()["session"]["id"]
    assert c.delete(f"/v1/sessions/{sid}").status_code == 200
    assert c.get(f"/v1/sessions/{sid}").status_code == 404
