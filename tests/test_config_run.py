"""Tests for the config-driven runner (the cluster entrypoint).

Uses the torch-free FakeBackend so the full load-config -> run -> report -> plot
path is exercised in CI without a model. The real models live behind the same
path via `backend: hf|vllm` in the config.
"""

from __future__ import annotations

import json
from pathlib import Path

from dexa.bench.config import RunConfig, NiahSuite, AgenticSuite, load_config


def test_load_config_roundtrip(tmp_path: Path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({
        "name": "t", "backend": "fake", "out_dir": str(tmp_path / "out"),
        "niah": {"lengths": [120], "ratios": [4, 16], "seeds": 1, "ref_strategy": "self"},
        "agentic": {"enabled": False},
    }))
    cfg = load_config(str(cfg_file))
    assert cfg.name == "t" and cfg.backend == "fake"
    assert cfg.niah.lengths == [120] and cfg.niah.ratios == [4, 16]
    assert cfg.agentic.enabled is False


def test_load_config_rejects_unknown_key(tmp_path: Path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({"name": "t", "bogus_key": 1}))
    try:
        load_config(str(cfg_file))
        assert False, "should have rejected unknown key"
    except ValueError as e:
        assert "bogus_key" in str(e)


def test_run_config_fake_end_to_end(tmp_path: Path):
    from dexa.bench.run import run_config

    cfg = RunConfig(
        name="fake-smoke", backend="fake", out_dir=str(tmp_path / "out"), plots=False,
        niah=NiahSuite(enabled=True, lengths=[120], ratios=[4, 16], seeds=1,
                       ref_strategy="self", n_ref=32),
        agentic=AgenticSuite(enabled=False),
    )
    results = run_config(cfg)
    assert "niah" in results
    out = Path(cfg.out_dir)
    assert (out / "niah.json").exists()
    assert (out / "results.json").exists()
    assert (out / "REPORT.md").exists()
    # Plumbing checks only: FakeBackend.score is a stub, so recall *values* are
    # not meaningful here (real quality needs a real model). Assert the suite ran
    # every cell and produced the expected structure.
    summ = results["niah"]["summary"]
    assert summ, "summary should be non-empty"
    for r in summ:
        assert "recall:attention_matching" in r and "recall:random_subset" in r
    rows = results["niah"]["rows"]
    assert any(r["method"] == "full_kv" and r["recall_frac"] == 1.0 for r in rows)
