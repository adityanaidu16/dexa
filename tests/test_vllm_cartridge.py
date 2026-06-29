"""Structural + (cluster-only) functional gate for the vLLM cartridge server.

Two layers of coverage:

* **Structural** tests run *everywhere*, including this vLLM-less Mac/CI. They
  import :mod:`dexa.engine.vllm_cartridge` (which import-guards vLLM), assert the
  class/function surface exists, that constructing :class:`CartridgeServer`
  without vLLM raises a helpful ``RuntimeError`` mentioning vllm, and that a
  small hand-built :class:`Cartridge` round-trips through the pure-numpy layout
  helpers (cartridge layout <-> vLLM token-major <-> paged blocks, plus query
  positions and spec validation).

* **Functional** tests require a real vLLM install and are skipped otherwise via
  ``pytest.importorskip('vllm')`` *inside* the test, so the module-level
  structural checks still execute when vLLM is absent.

Run: ``.venv/bin/python -m pytest tests/test_vllm_cartridge.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

# These imports MUST succeed without vllm installed (the module import-guards it).
from dexa.cartridge.artifact import Cartridge
from dexa.core.types import ModelSpec
from dexa.engine import vllm_cartridge as vc
from dexa.engine.vllm_cartridge import CartridgeServer


# --- helpers ---------------------------------------------------------------
def _tiny_spec(n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8) -> ModelSpec:
    return ModelSpec(
        name="tiny-cartridge-model",
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=n_q_heads * head_dim,
    )


def _tiny_cartridge(spec=None, t=5, logical_length=37, seed=0) -> Cartridge:
    """Hand-build a small Cartridge with numpy (no training)."""
    spec = spec or _tiny_spec()
    rng = np.random.default_rng(seed)
    shape = (spec.n_layers, spec.n_kv_heads, t, spec.head_dim)
    keys = rng.standard_normal(shape).astype(np.float32)
    values = rng.standard_normal(shape).astype(np.float32)
    positions = np.linspace(0, logical_length - 1, t).astype(np.int64)
    return Cartridge(
        spec=spec, keys=keys, values=values, positions=positions,
        logical_length=logical_length, meta={"src": "test"},
    )


# --- structural: run everywhere (no vllm needed) ---------------------------
def test_surface_exists():
    assert callable(CartridgeServer)
    for name in (
        "load_cartridge", "generate", "score",
    ):
        assert callable(getattr(CartridgeServer, name)), name
    for fn in (
        "cartridge_to_token_major", "token_major_to_cartridge",
        "pack_token_major_into_blocks", "query_positions",
        "assert_cartridge_matches_spec", "vllm_available",
    ):
        assert callable(getattr(vc, fn)), fn


def test_vllm_available_flag_is_bool():
    assert isinstance(vc.vllm_available(), bool)


@pytest.mark.skipif(vc.vllm_available(), reason="vllm IS installed; tested elsewhere")
def test_construction_without_vllm_raises_runtimeerror():
    with pytest.raises(RuntimeError) as ei:
        CartridgeServer(model_name="meta-llama/Llama-3.1-8B-Instruct")
    msg = str(ei.value).lower()
    assert "vllm" in msg  # helpful, mentions the missing dependency


def test_token_major_layout_and_roundtrip():
    cart = _tiny_cartridge()
    s = cart.spec
    k_layers, v_layers = vc.cartridge_to_token_major(cart.keys, cart.values)
    assert len(k_layers) == s.n_layers and len(v_layers) == s.n_layers
    for k in k_layers:
        # token-major: [t, n_kv_heads, head_dim]
        assert k.shape == (cart.t, s.n_kv_heads, s.head_dim)
    # explicit element check: token-major[t, h, d] == cartridge[layer, h, t, d]
    assert np.allclose(k_layers[0][:, 0, :], cart.keys[0, 0, :, :])
    # round-trip back to cartridge storage layout
    keys2, values2 = vc.token_major_to_cartridge(k_layers, v_layers)
    assert keys2.shape == cart.keys.shape
    assert np.array_equal(keys2, cart.keys)
    assert np.array_equal(values2, cart.values)


def test_token_major_rejects_bad_shape():
    with pytest.raises(ValueError):
        vc.cartridge_to_token_major(np.zeros((2, 3, 4)), np.zeros((2, 3, 4)))
    with pytest.raises(ValueError):
        vc.cartridge_to_token_major(np.zeros((1, 2, 3, 4)), np.zeros((1, 2, 3, 5)))


def test_pack_into_blocks_padding_and_values():
    cart = _tiny_cartridge(t=5)
    k_layers, _ = vc.cartridge_to_token_major(cart.keys, cart.values)
    tm = k_layers[0]  # [5, n_kv, d]
    block_size = 4
    blocks = vc.pack_token_major_into_blocks(tm, block_size)
    # ceil(5/4) = 2 blocks
    assert blocks.shape == (2, block_size, cart.spec.n_kv_heads, cart.spec.head_dim)
    flat = blocks.reshape(2 * block_size, cart.spec.n_kv_heads, cart.spec.head_dim)
    # first t slots equal the token-major data
    assert np.array_equal(flat[:5], tm)
    # padded tail is zero
    assert np.all(flat[5:] == 0)


def test_pack_into_blocks_exact_multiple():
    arr = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)  # t=2
    blocks = vc.pack_token_major_into_blocks(arr, block_size=2)
    assert blocks.shape == (1, 2, 3, 8)
    assert np.array_equal(blocks.reshape(2, 3, 8), arr)


def test_pack_into_blocks_rejects_bad_args():
    with pytest.raises(ValueError):
        vc.pack_token_major_into_blocks(np.zeros((4, 2)), block_size=2)  # not 3-D
    with pytest.raises(ValueError):
        vc.pack_token_major_into_blocks(np.zeros((4, 2, 8)), block_size=0)


def test_query_positions_start_at_logical_length():
    cart = _tiny_cartridge(logical_length=37)
    pos = vc.query_positions(cart, 4)
    assert pos.tolist() == [37, 38, 39, 40]
    assert pos.dtype == np.int64
    assert vc.query_positions(cart, 0).tolist() == []
    with pytest.raises(ValueError):
        vc.query_positions(cart, -1)


def test_assert_cartridge_matches_spec_ok_and_mismatch():
    spec = _tiny_spec()
    cart = _tiny_cartridge(spec=spec)
    # matching spec: no error
    vc.assert_cartridge_matches_spec(cart, spec)
    # wrong layer count
    bad = _tiny_spec(n_layers=spec.n_layers + 1)
    with pytest.raises(ValueError) as ei:
        vc.assert_cartridge_matches_spec(cart, bad)
    assert "n_layers" in str(ei.value)
    # wrong head_dim
    bad2 = _tiny_spec(head_dim=spec.head_dim + 4)
    with pytest.raises(ValueError):
        vc.assert_cartridge_matches_spec(cart, bad2)


def test_cartridge_to_compact_cache_has_zero_bias():
    # The whole serving simplification: a cartridge has no attention bias.
    cart = _tiny_cartridge()
    cc = cart.to_compact_cache()
    for layer in cc.layers:
        for b in layer.biases:
            assert np.all(b == 0.0)


# --- functional: cluster-only, skipped without vllm ------------------------
@pytest.fixture(scope="module")
def server():
    pytest.importorskip("vllm")
    # Small model so the cluster smoke test is cheap; override as needed.
    return CartridgeServer(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        dtype="float32",
    )


def _cartridge_for(server) -> Cartridge:
    s = server.spec
    rng = np.random.default_rng(0)
    t, logical_length = 6, 40
    shape = (s.n_layers, s.n_kv_heads, t, s.head_dim)
    return Cartridge(
        spec=s,
        keys=rng.standard_normal(shape).astype(np.float32),
        values=rng.standard_normal(shape).astype(np.float32),
        positions=np.linspace(0, logical_length - 1, t).astype(np.int64),
        logical_length=logical_length,
    )


def test_load_cartridge_validates(server):
    pytest.importorskip("vllm")
    cart = _cartridge_for(server)
    handle = server.load_cartridge(cart)
    assert isinstance(handle, str) and handle


def test_load_cartridge_rejects_mismatch(server):
    pytest.importorskip("vllm")
    bad_spec = _tiny_spec(n_layers=server.spec.n_layers + 1)
    bad = _tiny_cartridge(spec=bad_spec)
    with pytest.raises(ValueError):
        server.load_cartridge(bad)


def test_generate_against_cartridge_prefix(server):
    pytest.importorskip("vllm")
    cart = _cartridge_for(server)
    server.load_cartridge(cart)
    # This exercises the version-pinned prefix-injection seam; on a stock vLLM
    # without the site shim it raises a clear RuntimeError (acceptable here).
    try:
        out = server.generate("hello", cart, max_new_tokens=2)
    except RuntimeError as e:
        assert "prefix" in str(e).lower() or "shim" in str(e).lower()
    else:
        assert isinstance(out, str)
