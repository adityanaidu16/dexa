"""Smoke + correctness tests for the benchmark harness (FakeBackend only)."""

from __future__ import annotations

import json
import os

import pytest

from dexa.bench import _compactors
from dexa.bench.report import aggregate_rows, render_report, render_table
from dexa.bench.runner import run_matrix
from dexa.bench.tasks import make_tasks, niah_single
from dexa.engine.fake import FakeBackend

RATIOS = [2.0, 8.0]
COMPACTORS = ["full_kv", "random_subset", "recent_window"]
if "attention_matching" in _compactors.available_compactors():
    COMPACTORS.append("attention_matching")


@pytest.fixture(scope="module")
def backend():
    return FakeBackend()


@pytest.fixture(scope="module")
def tasks(backend):
    # small + short tasks: keep it fast but long enough for selection to matter
    return make_tasks(backend, lengths=[96], n_per=1, names=["niah_single", "multihop"])


def test_tasks_are_valid(backend):
    t = niah_single(backend, length=64)
    assert t.context_ids and t.prompt_ids and t.gold_ids
    assert all(isinstance(x, int) for x in t.context_ids)
    assert t.meta["value"]


def test_runner_produces_a_row_per_cell(backend, tasks, tmp_path):
    out = os.path.join(tmp_path, "results.json")
    result = run_matrix(
        backend, compactors=COMPACTORS, ratios=RATIOS, tasks=tasks, out_path=out
    )
    n_non_full = len([c for c in COMPACTORS if c != "full_kv"])
    # per task: 1 full_kv reference row + (compactor x ratio) cells
    expected = len(tasks) * (1 + n_non_full * len(RATIOS))
    assert len(result.rows) == expected
    # every cell has the metric keys
    for r in result.rows:
        assert "recon_rel_l2" in r and "compression_ratio" in r and "memory_saving" in r


def test_fullkv_recon_is_zero(backend, tasks):
    result = run_matrix(
        backend, compactors=COMPACTORS, ratios=RATIOS, tasks=tasks, out_path=None
    )
    full_rows = result.filter(compactor="full_kv")
    assert full_rows
    for r in full_rows:
        assert r["recon_rel_l2"] == pytest.approx(0.0, abs=1e-5)
        assert r["recon_cosine"] == pytest.approx(1.0, abs=1e-5)
        assert r["compression_ratio"] == pytest.approx(1.0)
        assert r["memory_saving"] == pytest.approx(0.0, abs=1e-9)


def test_results_json_written(backend, tasks, tmp_path):
    out = os.path.join(tmp_path, "results.json")
    run_matrix(backend, compactors=COMPACTORS, ratios=RATIOS, tasks=tasks, out_path=out)
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert "rows" in data and data["rows"]
    assert "cost_model" in data


def test_report_renders(backend, tasks, tmp_path):
    result = run_matrix(
        backend, compactors=COMPACTORS, ratios=RATIOS, tasks=tasks, out_path=None
    )
    agg = aggregate_rows(result.rows)
    assert agg
    text = render_table(agg)
    assert "compactor" in text
    # full render (table + plots) should not raise
    summary = render_report(result, out_dir=str(tmp_path), plots=True)
    assert summary["table"]


def test_compression_increases_memory_saving(backend, tasks):
    result = run_matrix(
        backend, compactors=["random_subset"], ratios=RATIOS, tasks=tasks, out_path=None
    )
    agg = {r["ratio"]: r for r in aggregate_rows(result.rows) if r["compactor"] == "random_subset"}
    assert agg[8.0]["memory_saving"] > agg[2.0]["memory_saving"]


@pytest.mark.skipif(
    "attention_matching" not in _compactors.available_compactors(),
    reason="attention_matching compactor not available",
)
def test_attention_matching_beats_random(backend, tasks):
    result = run_matrix(
        backend,
        compactors=["random_subset", "attention_matching"],
        ratios=RATIOS,
        tasks=tasks,
        out_path=None,
    )
    agg = aggregate_rows(result.rows)
    for ratio in RATIOS:
        am = [r for r in agg if r["compactor"] == "attention_matching" and r["ratio"] == ratio]
        rs = [r for r in agg if r["compactor"] == "random_subset" and r["ratio"] == ratio]
        assert am and rs
        assert am[0]["recon_rel_l2"] < rs[0]["recon_rel_l2"], (
            f"at ratio {ratio}: AM={am[0]['recon_rel_l2']} RS={rs[0]['recon_rel_l2']}"
        )
