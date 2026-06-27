"""Tests for the STILL amortized-perceiver compactor.

These are deliberately tiny and fast (synthetic tensors / the tiny-random Llama,
seconds on CPU). They check the contract that matters:

  (a) the perceiver forward produces correctly-shaped compact K/V/beta;
  (b) at identity init with ``t == T`` the compact cache reproduces the input KV
      (the identity property that makes training start from a sane place);
  (c) a few distillation steps on a tiny synthetic task reduce the KL loss
      (optimization actually works);
  (d) :meth:`StillCompactor.compact` returns a valid CompactCache.

Run: ``.venv/bin/python -m pytest tests/test_still.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dexa.compaction.base import CompactionBudget
from dexa.compaction.still.compactor import StillCompactor
from dexa.compaction.still.perceiver import StillPerceiver
from dexa.core.types import KVCache, LayerKV, ModelSpec

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _rope(x, positions, inv_freq, sign=1.0):
    """Reference RoPE (matches perceiver._rope) for building post-RoPE keys."""
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * emb.cos() + sign * rot * emb.sin()


def _synthetic_kv(n_kv=2, T=6, d=4, seed=0):
    """A KVCache with genuine post-RoPE keys (un-rotated rand keys re-rotated)."""
    g = torch.Generator().manual_seed(seed)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    positions = torch.arange(T).float()
    k_unrot = torch.randn(n_kv, T, d, generator=g)
    key = _rope(k_unrot, positions, inv_freq, sign=1.0)  # post-RoPE
    value = torch.randn(n_kv, T, d, generator=g)
    spec = ModelSpec(
        name="synthetic", n_layers=1, n_q_heads=n_kv, n_kv_heads=n_kv,
        head_dim=d, hidden_size=n_kv * d,
    )
    kv = KVCache(
        spec=spec,
        layers=[LayerKV(key=key.numpy().astype(np.float32),
                        value=value.numpy().astype(np.float32))],
        positions=np.arange(T, dtype=np.int64),
        token_ids=list(range(T)),
    )
    return kv, key, value, positions


# --- (a) forward shapes ---------------------------------------------------
@pytest.mark.torch
def test_perceiver_forward_shapes():
    n_kv, T, d, t = 3, 8, 4, 3
    _, key, value, positions = _synthetic_kv(n_kv=n_kv, T=T, d=d)
    perc = StillPerceiver(head_dim=d, n_latents=t)
    Ck, Cv, beta, pos = perc(key, value, positions)
    assert Ck.shape == (n_kv, t, d)
    assert Cv.shape == (n_kv, t, d)
    assert beta.shape == (n_kv, t)
    assert pos.shape == (t,)
    assert torch.isfinite(Ck).all() and torch.isfinite(Cv).all() and torch.isfinite(beta).all()


# --- (b) identity property at t == T -------------------------------------
@pytest.mark.torch
def test_identity_init_reconstructs_kv_when_t_equals_T():
    n_kv, T, d = 2, 6, 4
    _, key, value, positions = _synthetic_kv(n_kv=n_kv, T=T, d=d, seed=1)
    perc = StillPerceiver(head_dim=d, n_latents=T)  # t == T
    perc.eval()
    with torch.no_grad():
        Ck, Cv, beta, pos = perc(key, value, positions)

    # Compact keys/values reproduce the input (routing is one-hot on the
    # diagonal; un-rotate then re-rotate at the same positions is the identity).
    assert torch.allclose(Ck, key, atol=1e-3), f"key max diff {(Ck - key).abs().max()}"
    assert torch.allclose(Cv, value, atol=1e-3), f"value max diff {(Cv - value).abs().max()}"
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-4)
    assert torch.allclose(pos, positions, atol=1e-4)


# --- (c) optimization reduces KL -----------------------------------------
@pytest.mark.torch
def test_training_reduces_kl():
    from dexa.engine.hf_backend import HFBackend
    from dexa.compaction.still.train import build_perceivers, random_samples, train

    backend = HFBackend(model_name=MODEL, device="cpu", dtype="float32")
    # Compress (t < T) so there is a real reconstruction error to minimize.
    perceivers = build_perceivers(backend, n_latents=4)
    samples = random_samples(backend, 1, context_len=12, answer_len=4, seed=3)

    history = train(backend, samples, perceivers, steps=12, lr=2e-3)

    assert len(history) == 12
    assert all(np.isfinite(history))
    # Overfitting one fixed batch must drive the KL down.
    assert history[-1] < history[0], f"KL did not decrease: {history[0]} -> {history[-1]}"


# --- (d) compactor returns a valid CompactCache --------------------------
@pytest.mark.torch
def test_compactor_returns_valid_compact_cache():
    n_kv, T, d = 2, 10, 4
    kv, *_ = _synthetic_kv(n_kv=n_kv, T=T, d=d, seed=2)
    compactor = StillCompactor()
    assert compactor.name == "still"
    assert compactor.needs_ref_queries is False

    budget = CompactionBudget(tokens_per_head=3)
    compact = compactor.compact(kv, budget)

    assert compact.method == "still"
    assert compact.logical_length == T
    assert len(compact.layers) == kv.spec.n_layers
    for layer in compact.layers:
        assert len(layer.keys) == n_kv
        assert len(layer.values) == n_kv
        assert len(layer.biases) == n_kv
        for h in range(n_kv):
            assert layer.keys[h].shape == (3, d)
            assert layer.values[h].shape == (3, d)
            assert layer.biases[h].shape == (3,)
            assert np.isfinite(layer.keys[h]).all()
            assert np.isfinite(layer.values[h]).all()
    assert compact.budget == kv.spec.n_layers * n_kv * 3


# --- bonus: compactor identity at no-compression limit -------------------
@pytest.mark.torch
def test_compactor_identity_at_no_compression():
    n_kv, T, d = 2, 5, 4
    kv, key, value, _ = _synthetic_kv(n_kv=n_kv, T=T, d=d, seed=4)
    compactor = StillCompactor()
    compact = compactor.compact(kv, CompactionBudget(tokens_per_head=T))
    for h in range(n_kv):
        assert np.allclose(compact.layers[0].keys[h], key[h].numpy(), atol=1e-3)
        assert np.allclose(compact.layers[0].values[h], value[h].numpy(), atol=1e-3)
