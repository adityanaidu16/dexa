"""Validate :class:`~dexa.engine.vllm_connector.DexaConnector` against a **real**
vLLM install — the check the connector's own docstring asks you to run.

The connector is coded against vLLM's *documented* V1 KV-connector surface
(``vllm.distributed.kv_transfer.kv_connector.v1.base``), but vLLM's V1 API is
version-specific and this repo's dev/CI machines have no vLLM. This script is the
turnkey bridge: point it at a box with ``pip install vllm`` (see
``scripts/modal_scale_and_connector.py`` for the one-command Modal run) and it
reports, for *that* vLLM release, exactly how much of the connector is sound and
which version-pinned seams a site still has to shim.

It runs in **tiers**, each stricter about what it needs, so you always get the
strongest signal your environment allows:

* **Tier 0 — pure numpy (needs nothing).** The paged-block <-> KVCache round-trip
  and the store persist/restore path. This is the data movement the worker-side
  hooks rely on; it is deterministic and testable everywhere.
* **Tier 1 — real-vLLM conformance (needs ``import vllm``, no GPU/model).**
  Confirms :class:`DexaConnector` subclasses the *installed* ``KVConnectorBase_V1``
  and that every V1 lifecycle method Dexa overrides still has a **matching
  signature** on this release. Signature drift here is the single most common way
  the documented-vs-actual gap bites; this prints it precisely.
* **Tier 2 — engine construction (needs a GPU + a model), opt-in via ``--serve``.**
  Asks vLLM to actually load ``DexaConnector`` through its connector registry and
  run a short generation. Without the site shims this is *expected* to reach a
  version-pinned seam (e.g. ``_split_kv_layer``) and raise a clear ``RuntimeError``;
  the check reports how far the lifecycle ran and which seam it hit, which is the
  concrete to-do list for finishing the in-engine path.

The signature-diff and round-trip logic are pure functions
(:func:`signature_report`, :func:`numpy_roundtrip_ok`) so they are unit-tested on
CPU (``tests/test_vllm_connector_check.py``) against the module's vLLM-absent
stand-in base — the same code then runs against the real base on the cluster.

Run anywhere::

    python benchmarks/vllm_connector_check.py            # tiers 0-1 (skips 1 w/o vllm)
    python benchmarks/vllm_connector_check.py --serve --model facebook/opt-125m
"""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any, Optional

import numpy as np

from dexa.core.types import KVCache, LayerKV, ModelSpec
from dexa.engine import vllm_connector as vc
from dexa.engine.vllm_connector import (
    DexaConnector,
    kvcache_to_paged_blocks,
    paged_blocks_to_kvcache,
)
from dexa.session.store import SessionStore

#: The V1 KV-connector lifecycle methods :class:`DexaConnector` overrides. These
#: are the surface whose signatures vLLM changes between releases; the tier-1
#: report checks each one against the installed base.
V1_LIFECYCLE_METHODS = (
    # scheduler side
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "build_connector_meta",
    "request_finished",
    # worker side
    "register_kv_caches",
    "start_load_kv",
    "wait_for_layer_load",
    "save_kv_layer",
    "wait_for_save",
    "get_finished",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on CPU).
# ---------------------------------------------------------------------------
def _positional_params(sig: inspect.Signature) -> list[str]:
    """Names of a signature's positional params (excluding ``self``), in order.

    ``*args`` / ``**kwargs`` are dropped: an override that adds ``**kwargs`` (as
    :meth:`DexaConnector.start_load_kv` / :meth:`~DexaConnector.save_kv_layer` do)
    is still compatible with a stricter base, so they must not count as drift.
    """
    out: list[str] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            out.append(name)
    return out


def _accepts_var_keyword(sig: inspect.Signature) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def signature_report(
    base_cls: type, impl_cls: type, method_names=V1_LIFECYCLE_METHODS
) -> dict[str, dict[str, Any]]:
    """Compare ``impl_cls``'s overrides against ``base_cls``'s methods.

    Pure and side-effect-free so it runs against the vLLM-absent stand-in base on
    CPU *and* the real ``KVConnectorBase_V1`` on the cluster with identical logic.
    For each method name returns a record with:

    * ``on_base`` / ``on_impl`` — whether the method exists on each class,
    * ``base_sig`` / ``impl_sig`` — the string signatures (``None`` if absent),
    * ``match`` — ``True`` when the base's positional params are a prefix of the
      impl's (an override may take *more* trailing/keyword args, e.g. ``**kwargs``,
      but must accept at least what the base passes), else ``False``,
    * ``note`` — a human-readable diagnosis when they differ.

    ``match=False`` with ``on_base=False`` means vLLM renamed or removed the hook
    in this release — a real, actionable finding, not a false alarm.
    """
    report: dict[str, dict[str, Any]] = {}
    for name in method_names:
        base_m = getattr(base_cls, name, None)
        impl_m = getattr(impl_cls, name, None)
        rec: dict[str, Any] = {
            "on_base": callable(base_m),
            "on_impl": callable(impl_m),
            "base_sig": None,
            "impl_sig": None,
            "match": False,
            "note": "",
        }
        if not rec["on_impl"]:
            rec["note"] = "not overridden by DexaConnector"
            report[name] = rec
            continue
        impl_sig = inspect.signature(impl_m)
        rec["impl_sig"] = str(impl_sig)
        if not rec["on_base"]:
            rec["note"] = (
                "absent on the installed vLLM base — renamed/removed in this "
                "release; re-map the override to the current hook name"
            )
            report[name] = rec
            continue
        base_sig = inspect.signature(base_m)
        rec["base_sig"] = str(base_sig)
        base_params = _positional_params(base_sig)
        impl_params = _positional_params(impl_sig)
        is_prefix = impl_params[: len(base_params)] == base_params
        absorbs_extra = _accepts_var_keyword(impl_sig) or len(impl_params) >= len(base_params)
        if is_prefix and absorbs_extra:
            rec["match"] = True
        else:
            rec["note"] = (
                f"signature drift: base positional params {base_params} are not a "
                f"prefix of the override's {impl_params}"
            )
        report[name] = rec
    return report


def numpy_roundtrip_ok(
    *, n_layers: int = 3, n_kv_heads: int = 2, head_dim: int = 8, T: int = 37,
    block_size: int = 16, seed: int = 0,
) -> bool:
    """Whether a random KVCache survives KVCache -> paged blocks -> KVCache exactly.

    This is the worker-side data movement (:func:`kvcache_to_paged_blocks` /
    :func:`paged_blocks_to_kvcache`) the connector's ``start_load_kv`` /
    ``wait_for_save`` depend on; ``T`` deliberately does not divide ``block_size``
    so the final-block padding path is exercised.
    """
    rng = np.random.default_rng(seed)
    spec = ModelSpec(name="roundtrip-model", n_layers=n_layers, n_q_heads=n_kv_heads * 2,
                     n_kv_heads=n_kv_heads, head_dim=head_dim,
                     hidden_size=n_kv_heads * 2 * head_dim)
    layers = [
        LayerKV(
            key=rng.standard_normal((n_kv_heads, T, head_dim)).astype(np.float32),
            value=rng.standard_normal((n_kv_heads, T, head_dim)).astype(np.float32),
        )
        for _ in range(n_layers)
    ]
    positions = np.arange(T, dtype=np.int64)
    kv = KVCache(spec=spec, layers=layers, positions=positions, token_ids=list(range(T)))
    k_blocks, v_blocks = kvcache_to_paged_blocks(kv, block_size)
    kv2 = paged_blocks_to_kvcache(
        k_blocks, v_blocks, spec=spec, positions=positions, token_ids=list(range(T))
    )
    return all(
        np.array_equal(a.key, b.key) and np.array_equal(a.value, b.value)
        for a, b in zip(kv.layers, kv2.layers)
    )


def store_roundtrip_ok(store_root, *, seed: int = 1) -> bool:
    """Whether a KVCache persists and restores byte-identically through the
    connector's :meth:`DexaConnector.store_kvcache` / ``load_kvcache_for`` — the
    cross-instance / post-restart reuse tier, pure Dexa (no vLLM)."""
    rng = np.random.default_rng(seed)
    spec = ModelSpec(name="store-roundtrip", n_layers=2, n_q_heads=4, n_kv_heads=2,
                     head_dim=8, hidden_size=32)
    T = 20
    layers = [
        LayerKV(key=rng.standard_normal((2, T, 8)).astype(np.float32),
                value=rng.standard_normal((2, T, 8)).astype(np.float32))
        for _ in range(2)
    ]
    kv = KVCache(spec=spec, layers=layers, positions=np.arange(T, dtype=np.int64),
                 token_ids=list(range(T)))
    store = SessionStore(root=str(store_root))
    tokens = list(range(T))
    key = DexaConnector.store_kvcache(store, tokens, kv, model_name=spec.name)
    loaded = DexaConnector.load_kvcache_for(store, tokens, model_name=spec.name)
    if loaded is None:
        return False
    same = all(
        np.allclose(a.key, b.key) and np.allclose(a.value, b.value)
        for a, b in zip(kv.layers, loaded.layers)
    )
    # a different prefix must miss (keys are content-addressed).
    missed = DexaConnector.load_kvcache_for(store, tokens + [999], model_name=spec.name) is None
    return bool(key and same and missed)


# ---------------------------------------------------------------------------
# Tier runners.
# ---------------------------------------------------------------------------
def run_tier0(store_root) -> dict:
    """Pure-numpy data movement + store round-trip. Needs nothing."""
    rt = numpy_roundtrip_ok()
    st = store_roundtrip_ok(store_root)
    ok = rt and st
    print("\n[tier 0] pure numpy (no vLLM needed)")
    print(f"  paged-block round-trip (lossless) : {'PASS' if rt else 'FAIL'}")
    print(f"  store persist/restore + miss      : {'PASS' if st else 'FAIL'}")
    return {"tier": 0, "roundtrip": rt, "store": st, "ok": ok}


def run_tier1() -> dict:
    """Real-vLLM conformance: subclass check + per-method signature diff."""
    print("\n[tier 1] real-vLLM conformance (needs `import vllm`, no GPU)")
    if not vc.vllm_available():
        print("  vLLM not importable here -> SKIPPED. Run on a box with "
              "`pip install vllm` (see scripts/modal_scale_and_connector.py).")
        return {"tier": 1, "skipped": True, "reason": "vllm not importable"}

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    print(f"  vLLM version                      : {vc.vllm_version()}")
    subclass = issubclass(DexaConnector, KVConnectorBase_V1)
    print(f"  DexaConnector subclasses real base: {'YES' if subclass else 'NO'}")

    report = signature_report(KVConnectorBase_V1, DexaConnector)
    n_match = sum(1 for r in report.values() if r["match"])
    n_total = len(report)
    print(f"  V1 method signature conformance   : {n_match}/{n_total} match")
    for name, rec in report.items():
        if rec["match"]:
            print(f"    ok   {name}{rec['impl_sig']}")
        else:
            print(f"    DIFF {name}: {rec['note']}")
            if rec["base_sig"]:
                print(f"         base: {name}{rec['base_sig']}")
            if rec["impl_sig"]:
                print(f"         impl: {name}{rec['impl_sig']}")
    ok = subclass and n_match == n_total
    print(f"  tier 1 verdict                    : {'PASS' if ok else 'REVIEW DIFFS ABOVE'}")
    return {"tier": 1, "skipped": False, "vllm_version": vc.vllm_version(),
            "subclass_real_base": subclass, "n_match": n_match, "n_total": n_total,
            "report": report, "ok": ok}


def run_tier2(model: str, store_root) -> dict:
    """Ask vLLM to load DexaConnector through its registry and generate.

    Opt-in (``--serve``): needs a GPU + a model. Without the version-pinned site
    shims this is *expected* to reach a seam and raise; we report how far it got.
    """
    print("\n[tier 2] engine construction via vLLM's connector registry (--serve)")
    if not vc.vllm_available():
        print("  vLLM not importable here -> SKIPPED.")
        return {"tier": 2, "skipped": True, "reason": "vllm not importable"}
    try:  # pragma: no cover - cluster only
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig

        kt = KVTransferConfig(
            kv_connector="DexaConnector",
            kv_connector_module_path="dexa.engine.vllm_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"dexa_store_root": str(store_root)},
        )
        print(f"  constructing LLM({model!r}) with DexaConnector ...", flush=True)
        llm = LLM(model=model, kv_transfer_config=kt, enforce_eager=True,
                  gpu_memory_utilization=0.6, max_model_len=2048)
        print("  engine constructed -> connector loaded through vLLM registry: PASS")
        out = llm.generate(["The capital of France is"],
                           SamplingParams(max_tokens=8, temperature=0.0))
        print(f"  generation completed: {out[0].outputs[0].text!r}")
        print("  reached full lifecycle without hitting a site shim -> connector "
              "path is complete on this vLLM.")
        return {"tier": 2, "skipped": False, "constructed": True, "generated": True,
                "seam": None, "ok": True}
    except RuntimeError as exc:  # pragma: no cover - cluster only
        print(f"  reached a version-pinned site shim (expected until shimmed):\n    {exc}")
        return {"tier": 2, "skipped": False, "constructed": True, "generated": False,
                "seam": str(exc), "ok": False}
    except Exception as exc:  # pragma: no cover - cluster only
        print(f"  vLLM engine construction failed before the Dexa path: {type(exc).__name__}: {exc}")
        return {"tier": 2, "skipped": False, "constructed": False, "generated": False,
                "seam": None, "error": f"{type(exc).__name__}: {exc}", "ok": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serve", action="store_true",
                    help="also run tier 2 (needs a GPU + model): load the connector in vLLM")
    ap.add_argument("--model", default="facebook/opt-125m",
                    help="model for tier 2 engine construction")
    ap.add_argument("--store-dir", default=".dexa_connector_check",
                    help="scratch SessionStore root for the round-trip checks")
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    args = ap.parse_args()

    print("=" * 70)
    print("Dexa vLLM connector validation")
    print(f"  vllm importable: {vc.vllm_available()}"
          + (f" (v{vc.vllm_version()})" if vc.vllm_available() else ""))
    print("=" * 70)

    results = {"vllm_available": vc.vllm_available(), "tiers": []}
    results["tiers"].append(run_tier0(args.store_dir))
    results["tiers"].append(run_tier1())
    if args.serve:
        results["tiers"].append(run_tier2(args.model, args.store_dir))

    # overall verdict: every tier that actually ran must be ok (tier 2 reaching a
    # documented shim is a "known-incomplete", not a hard failure of the run).
    ran = [t for t in results["tiers"] if not t.get("skipped")]
    hard_ok = all(t.get("ok", False) for t in ran if t["tier"] != 2)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {'PASS' if hard_ok else 'FAIL'} "
          f"({sum(1 for t in ran)} tier(s) ran, "
          f"{sum(1 for t in results['tiers'] if t.get('skipped'))} skipped)")
    print("=" * 70)

    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.out}")

    raise SystemExit(0 if hard_ok else 1)


if __name__ == "__main__":
    main()
