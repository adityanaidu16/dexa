"""Structural + (cluster-only) functional gate for the Dexa vLLM KV-connector.

Two layers of coverage, matching the other vLLM adapters
(:mod:`tests.test_vllm_backend`, :mod:`tests.test_vllm_cartridge`):

* **Structural** tests run *everywhere*, including this vLLM-less Mac/CI. They
  import :mod:`dexa.engine.vllm_connector` (which import-guards vLLM), assert the
  :class:`DexaConnector` class + V1 method surface exist and are introspectable,
  that constructing it without vLLM raises a helpful ``RuntimeError`` mentioning
  vllm (the same choice :class:`VLLMBackend` makes), that the pure-numpy helpers
  are deterministic / collision-resistant / round-trip, and that a small KVCache
  round-trips through the connector's :class:`SessionStore` persistence helpers.

* **Functional** tests require a real vLLM install and are skipped otherwise via
  ``pytest.importorskip('vllm')``, so the structural checks still run when vLLM
  is absent.

Run: ``.venv/bin/python -m pytest tests/test_vllm_connector.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

# These imports MUST succeed without vllm installed (the module import-guards it).
from dexa.core.types import KVCache, LayerKV, ModelSpec
from dexa.engine import vllm_connector as vc
from dexa.engine.vllm_connector import DexaConnector
from dexa.session.store import SessionStore


# --- helpers ---------------------------------------------------------------
def _tiny_spec(n_layers=3, n_q_heads=4, n_kv_heads=2, head_dim=8) -> ModelSpec:
    return ModelSpec(
        name="tiny-connector-model",
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=n_q_heads * head_dim,
    )


def _tiny_kvcache(spec=None, T=7, seed=0) -> KVCache:
    """Hand-build a small KVCache with numpy (no model)."""
    spec = spec or _tiny_spec()
    rng = np.random.default_rng(seed)
    layers = [
        LayerKV(
            key=rng.standard_normal((spec.n_kv_heads, T, spec.head_dim)).astype(np.float32),
            value=rng.standard_normal((spec.n_kv_heads, T, spec.head_dim)).astype(np.float32),
        )
        for _ in range(spec.n_layers)
    ]
    return KVCache(
        spec=spec,
        layers=layers,
        positions=np.arange(T, dtype=np.int64),
        token_ids=list(range(100, 100 + T)),
        meta={"src": "test"},
    )


# --- structural: run everywhere (no vllm needed) ---------------------------
def test_surface_exists():
    assert callable(DexaConnector)
    # V1 connector lifecycle: scheduler-side + worker-side hooks all present.
    for name in (
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "build_connector_meta",
        "request_finished",
        "register_kv_caches",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
        "wait_for_save",
        "get_finished",
        # pure-python persistence helpers
        "store_kvcache",
        "load_kvcache_for",
        "has_prefix",
    ):
        assert callable(getattr(DexaConnector, name)), name
    for fn in (
        "prefix_key",
        "kvcache_to_paged_blocks",
        "paged_blocks_to_kvcache",
        "vllm_available",
        "vllm_version",
    ):
        assert callable(getattr(vc, fn)), fn


def test_vllm_available_flag_is_bool_and_version():
    assert isinstance(vc.vllm_available(), bool)
    v = vc.vllm_version()
    assert v is None or isinstance(v, str)


@pytest.mark.skipif(vc.vllm_available(), reason="vllm IS installed; tested elsewhere")
def test_construction_without_vllm_raises_runtimeerror():
    with pytest.raises(RuntimeError) as ei:
        DexaConnector()
    msg = str(ei.value).lower()
    assert "vllm" in msg  # helpful, mentions the missing dependency
    # onboarding flag echoed so the error is actionable
    assert "kv-transfer-config" in msg or "kv_connector" in msg


# --- prefix_key: determinism + collision-resistance ------------------------
def test_prefix_key_is_deterministic():
    assert vc.prefix_key([1, 2, 3, 4]) == vc.prefix_key([1, 2, 3, 4])
    assert vc.prefix_key([]) == vc.prefix_key([])


def test_prefix_key_distinguishes_content_order_and_length():
    base = vc.prefix_key([1, 2, 3])
    assert base != vc.prefix_key([1, 2, 4])      # one token differs
    assert base != vc.prefix_key([3, 2, 1])      # order differs
    assert base != vc.prefix_key([1, 2, 3, 4])   # length differs (prefix vs extension)
    assert base != vc.prefix_key([1, 2])         # shorter prefix


def test_prefix_key_namespaces_by_model_and_is_fs_safe():
    a = vc.prefix_key([1, 2, 3], model_name="meta-llama/Llama-3.1-8B")
    b = vc.prefix_key([1, 2, 3], model_name="mistralai/Mistral-7B")
    assert a != b                                 # same tokens, different model
    assert a != vc.prefix_key([1, 2, 3])          # model-namespaced != bare
    # usable as a filename stem (no path separators / odd chars)
    for ch in "/\\: ":
        assert ch not in a


def test_prefix_key_low_collision_over_many_sequences():
    keys = {vc.prefix_key(list(range(i, i + 5))) for i in range(2000)}
    assert len(keys) == 2000  # no collisions across 2000 distinct prefixes


# --- paged-block <-> KVCache layout round-trip -----------------------------
@pytest.mark.parametrize("T,block_size", [(7, 4), (8, 4), (1, 16), (16, 16), (5, 1)])
def test_paged_block_layout_roundtrip(T, block_size):
    kv = _tiny_kvcache(T=T)
    k_blocks, v_blocks = vc.kvcache_to_paged_blocks(kv, block_size)
    assert len(k_blocks) == kv.spec.n_layers
    num_blocks = (T + block_size - 1) // block_size
    for kb in k_blocks:
        assert kb.shape == (num_blocks, block_size, kv.spec.n_kv_heads, kv.spec.head_dim)
    kv2 = vc.paged_blocks_to_kvcache(
        k_blocks, v_blocks, spec=kv.spec, positions=kv.positions, token_ids=kv.token_ids
    )
    assert len(kv2.layers) == kv.spec.n_layers
    for l0, l1 in zip(kv.layers, kv2.layers):
        assert l1.key.shape == l0.key.shape
        assert np.array_equal(l0.key, l1.key)
        assert np.array_equal(l0.value, l1.value)
    assert np.array_equal(kv2.positions, kv.positions)
    assert kv2.token_ids == kv.token_ids


def test_paged_blocks_padding_is_zero():
    kv = _tiny_kvcache(T=5)
    k_blocks, _ = vc.kvcache_to_paged_blocks(kv, block_size=4)
    # 5 tokens -> 2 blocks of 4; slots 5..7 are padding.
    flat = k_blocks[0].reshape(2 * 4, kv.spec.n_kv_heads, kv.spec.head_dim)
    assert np.all(flat[5:] == 0)


def test_kvcache_to_paged_blocks_rejects_bad_args():
    kv = _tiny_kvcache()
    with pytest.raises(ValueError):
        vc.kvcache_to_paged_blocks(kv, 0)
    # bad per-layer ndim
    bad = KVCache(
        spec=kv.spec,
        layers=[LayerKV(key=np.zeros((2, 8)), value=np.zeros((2, 8)))],
        positions=np.arange(8),
    )
    with pytest.raises(ValueError):
        vc.kvcache_to_paged_blocks(bad, 4)


def test_paged_blocks_to_kvcache_rejects_mismatched_lengths():
    kv = _tiny_kvcache()
    k_blocks, v_blocks = vc.kvcache_to_paged_blocks(kv, 4)
    with pytest.raises(ValueError):
        vc.paged_blocks_to_kvcache(
            k_blocks, v_blocks[:-1], spec=kv.spec, positions=kv.positions
        )


# --- SessionStore round-trip through the connector helpers -----------------
def test_store_roundtrip_via_connector(tmp_path):
    store = SessionStore(root=tmp_path / "kv")
    kv = _tiny_kvcache(T=9, seed=3)
    tokens = kv.token_ids

    # miss before save
    assert DexaConnector.load_kvcache_for(store, tokens) is None
    assert not DexaConnector.has_prefix(store, tokens)

    key = DexaConnector.store_kvcache(store, tokens, kv)
    assert isinstance(key, str) and key
    assert DexaConnector.has_prefix(store, tokens)

    # an identical prefix (e.g. another instance / after restart) loads it back,
    # bit-identical (the persistence guarantee that avoids re-prefill).
    got = DexaConnector.load_kvcache_for(store, tokens)
    assert got is not None
    for l0, l1 in zip(kv.layers, got.layers):
        assert np.array_equal(l0.key, l1.key)
        assert np.array_equal(l0.value, l1.value)
    assert np.array_equal(got.positions, kv.positions)

    # a different prefix is a miss
    assert DexaConnector.load_kvcache_for(store, [1, 2, 3]) is None


def test_store_key_matches_prefix_key(tmp_path):
    store = SessionStore(root=tmp_path / "kv")
    kv = _tiny_kvcache()
    tokens = [11, 22, 33]
    key = DexaConnector.store_kvcache(store, tokens, kv, model_name="m/x")
    assert key == vc.prefix_key(tokens, model_name="m/x")
    assert store.has(key)


# --- functional: cluster-only, skipped without vllm ------------------------
def test_real_v1_base_is_subclassed_when_vllm_present():
    pytest.importorskip("vllm")
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    assert issubclass(DexaConnector, KVConnectorBase_V1)
    # if vllm is importable we should have captured its version
    assert isinstance(vc.vllm_version(), str)


def test_load_decision_adaptive_crossover():
    from dexa.engine.vllm_connector import load_decision
    # adaptive: below the crossover -> don't load (re-prefill is cheaper); at/above -> load
    assert load_decision(8192, min_load_tokens=32768) is False
    assert load_decision(32768, min_load_tokens=32768) is True
    assert load_decision(65536, min_load_tokens=32768) is True


def test_load_decision_policies_and_contention():
    from dexa.engine.vllm_connector import load_decision
    # explicit policies override the cost check
    assert load_decision(1, policy="always", min_load_tokens=32768) is True
    assert load_decision(10**9, policy="never", min_load_tokens=32768) is False
    # contention lowers the crossover: GPU busy (factor 0.25) -> load at 8k
    assert load_decision(8192, min_load_tokens=32768, contention_factor=0.25) is True
    assert load_decision(8192, min_load_tokens=32768, contention_factor=1.0) is False
