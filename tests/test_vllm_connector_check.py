"""CPU coverage for ``benchmarks/vllm_connector_check.py``.

The real-vLLM tiers of that script (tier 1 signature conformance, tier 2 engine
construction) only run on a GPU box with vLLM installed, but their *logic* — the
signature diff and the pure-numpy / store round-trips — is provider-free and
tested here against synthetic bases, exactly the way the connector's own
structural tests exercise the vLLM-absent path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the benchmark script as a module (benchmarks/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "vllm_connector_check",
    Path(__file__).resolve().parents[1] / "benchmarks" / "vllm_connector_check.py",
)
cc = importlib.util.module_from_spec(_SPEC)
sys.modules["vllm_connector_check"] = cc
_SPEC.loader.exec_module(cc)

from dexa.engine.vllm_connector import DexaConnector  # noqa: E402
from dexa.engine import vllm_connector as vc  # noqa: E402


# --- pure round-trips ------------------------------------------------------
def test_numpy_roundtrip_ok_lossless():
    assert cc.numpy_roundtrip_ok() is True


def test_numpy_roundtrip_various_geometries():
    # T not divisible by block_size exercises the final-block padding path.
    assert cc.numpy_roundtrip_ok(T=1, block_size=8)
    assert cc.numpy_roundtrip_ok(T=64, block_size=16)
    assert cc.numpy_roundtrip_ok(T=100, block_size=16, n_layers=1, n_kv_heads=1)


def test_store_roundtrip_ok(tmp_path):
    assert cc.store_roundtrip_ok(tmp_path) is True


# --- signature_report ------------------------------------------------------
def test_signature_report_missing_on_stand_in_base():
    """Against the module's vLLM-absent stand-in base (which has none of the V1
    lifecycle hooks), every override is flagged as absent-on-base — the exact
    finding a site would get if their vLLM renamed the hooks."""
    report = cc.signature_report(vc.KVConnectorBase_V1, DexaConnector)
    assert set(report) == set(cc.V1_LIFECYCLE_METHODS)
    for name, rec in report.items():
        assert rec["on_impl"] is True, name
        assert rec["on_base"] is False, name
        assert rec["match"] is False, name
        assert "absent" in rec["note"]


def test_signature_report_matches_identical_base():
    """A base whose method signatures equal DexaConnector's overrides -> all match."""

    class FakeBase:
        def get_num_new_matched_tokens(self, request, num_computed_tokens): ...
        def update_state_after_alloc(self, request, blocks, num_external_tokens): ...
        def build_connector_meta(self, scheduler_output): ...
        def request_finished(self, request, block_ids): ...
        def register_kv_caches(self, kv_caches): ...
        def start_load_kv(self, forward_context, **kwargs): ...
        def wait_for_layer_load(self, layer_name): ...
        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs): ...
        def wait_for_save(self): ...
        def get_finished(self, finished_req_ids): ...

    report = cc.signature_report(FakeBase, DexaConnector)
    assert all(rec["match"] for rec in report.values()), {
        n: r["note"] for n, r in report.items() if not r["match"]
    }


def test_signature_report_detects_drift():
    """A base whose hook takes an extra positional param the override lacks is
    flagged as drift (the override could not accept what vLLM would pass)."""

    class DriftBase:
        # extra 'num_tokens' the override doesn't take -> not a prefix.
        def get_num_new_matched_tokens(self, request, num_computed_tokens, num_tokens): ...

    report = cc.signature_report(
        DriftBase, DexaConnector, method_names=["get_num_new_matched_tokens"]
    )
    rec = report["get_num_new_matched_tokens"]
    assert rec["on_base"] and rec["on_impl"]
    assert rec["match"] is False
    assert "drift" in rec["note"]


def test_signature_report_override_may_add_kwargs():
    """An override that adds **kwargs over a stricter base still matches — adding
    trailing keyword args is backward compatible."""

    class Base:
        def start_load_kv(self, forward_context): ...

    report = cc.signature_report(Base, DexaConnector, method_names=["start_load_kv"])
    assert report["start_load_kv"]["match"] is True


def test_signature_report_flags_unoverridden_method():
    class Base:
        def some_new_hook(self): ...

    report = cc.signature_report(Base, DexaConnector, method_names=["some_new_hook"])
    rec = report["some_new_hook"]
    assert rec["on_impl"] is False
    assert rec["match"] is False
    assert "not overridden" in rec["note"]


# --- tier runners (CPU paths) ----------------------------------------------
def test_run_tier0_passes(tmp_path):
    res = cc.run_tier0(tmp_path)
    assert res["ok"] is True
    assert res["roundtrip"] and res["store"]


def test_run_tier1_skips_without_vllm():
    res = cc.run_tier1()
    if vc.vllm_available():  # pragma: no cover - cluster only
        assert res["skipped"] is False
        assert "report" in res
    else:
        assert res["skipped"] is True
        assert "vllm" in res["reason"]
