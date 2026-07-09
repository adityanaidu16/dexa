"""Exact RoPE re-phasing on the real HF backend (Phase 1, Layer C generalization).

The position-only exactness gate. The same tokens prefilled at ``position_offset=0``
vs ``position_offset=delta`` attend identically (RoPE attention depends on *relative*
positions, unchanged by a rigid offset), so their hidden states — hence k_raw and V
— are identical; the keys differ *only* by the RoPE rotation R(delta). Therefore
re-phasing the offset-0 keys by delta must reconstruct the offset-delta keys, and
the values must match unchanged. If the rotation basis/convention were wrong, this
diverges.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dexa.engine.hf_backend import HFBackend  # noqa: E402
from dexa.segment.selective import rope_rephase_keys  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def be():
    return HFBackend(model_name=MODEL, device="cpu", dtype="float32")


@pytest.mark.parametrize("delta", [1, 5, 37])
def test_rephase_reconstructs_shifted_keys(be, delta):
    toks = be.tokenize("The tool returned a large JSON payload with several fields.")
    kv0 = be.prefill(toks, position_offset=0)
    kvd = be.prefill(toks, position_offset=delta)

    cos, sin = be.rephase_cos_sin(delta)
    for l0, ld in zip(kv0.layers, kvd.layers):
        rk = rope_rephase_keys(l0.key, cos, sin)
        assert np.allclose(rk, ld.key, atol=1e-5, rtol=1e-4)     # keys reconstructed
        assert np.allclose(l0.value, ld.value, atol=1e-6)         # values position-free


def test_rephase_zero_delta_is_identity(be):
    toks = be.tokenize("Hello world.")
    kv = be.prefill(toks)
    cos, sin = be.rephase_cos_sin(0)
    for l in kv.layers:
        assert np.allclose(rope_rephase_keys(l.key, cos, sin), l.key, atol=1e-6)
