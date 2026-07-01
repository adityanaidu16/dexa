"""Plumbing test for the accuracy-vs-KV-memory frontier (docs/BENCHMARK.md).

Runs the full method sweep on the tiny model (gibberish outputs, so accuracy is
~0 — this checks the apparatus produces well-formed frontier points, memory
ordering, and a verdict; the real numbers come from a real model on GPU).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dexa.engine.hf_backend import HFBackend
from dexa.bench.datasets import load_ruler
from dexa.bench.frontier import run_frontier


@pytest.mark.torch
def test_frontier_sweep_plumbing():
    be = HFBackend("hf-internal-testing/tiny-random-LlamaForCausalLM", device="cpu", dtype="float32")
    ex = load_ruler(task="niah_single", length=200, n=1)
    res = run_frontier(
        be, ex,
        methods=("full_context", "rag", "attention_matching", "heavy_hitter", "snapkv", "cartridge"),
        ratios=(8,), rag_ks=(1,), n_ref=16, cartridge_opts={"steps": 1},
        max_new_tokens=3, verbose=False,
    )
    pts = {(p["method"], p["setting"]) for p in res["points"]}
    # every method contributed at least one point
    for m in ("full_context", "rag", "attention_matching", "heavy_hitter", "snapkv", "cartridge"):
        assert any(pm == m for (pm, _) in pts), f"no frontier point for {m}"
    # full_context holds the most memory; compressed methods hold less
    mem = {p["method"]: p["memory_bytes"] for p in res["points"]}
    assert mem["full_context"] >= mem["attention_matching"]
    # verdict is well-formed
    assert "passes" in res["verdict"] and "detail" in res["verdict"]
    for p in res["points"]:
        assert 0.0 <= p["accuracy_f1"] <= 1.0 and p["memory_bytes"] > 0
