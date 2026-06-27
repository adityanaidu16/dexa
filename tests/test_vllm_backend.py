"""Structural + (cluster-only) functional gate for the vLLM backend.

Two layers of coverage:

* **Structural** tests run *everywhere*, including this vLLM-less Mac/CI. They
  import :mod:`dexa.engine.vllm_backend` (which must import-guard vLLM), assert
  the ABC/contract surface (``VLLMBackend`` is a concrete ``ModelBackend`` with
  all methods present and no remaining abstractmethods), and check that
  constructing without vLLM raises a helpful ``RuntimeError``.

* **Functional** tests require a real vLLM install and are skipped otherwise via
  ``pytest.importorskip('vllm')`` *inside* the test, so the module-level
  structural checks still execute when vLLM is absent.

Run: ``.venv/bin/python -m pytest tests/test_vllm_backend.py -v``
"""

from __future__ import annotations

import inspect

import pytest

# These imports MUST succeed without vllm installed (the module import-guards it).
from dexa.engine import vllm_backend as vb
from dexa.engine.base import ModelBackend
from dexa.engine.vllm_backend import VLLMBackend

# Methods every ModelBackend must expose.
_CONTRACT_METHODS = (
    "tokenize",
    "detokenize",
    "prefill",
    "reference_queries",
    "generate",
    "score",
    "attention_outputs",
)


# --- structural: run everywhere (no vllm needed) ---------------------------
def test_is_model_backend_subclass():
    assert issubclass(VLLMBackend, ModelBackend)


def test_is_concrete_no_abstract_methods():
    # If any abstractmethod were left unimplemented, instantiation would be
    # impossible and this set would be non-empty.
    assert VLLMBackend.__abstractmethods__ == frozenset()


def test_contract_methods_exist():
    for name in _CONTRACT_METHODS:
        assert callable(getattr(VLLMBackend, name)), name
    # ``spec`` is a property on the ABC.
    assert isinstance(inspect.getattr_static(VLLMBackend, "spec"), property)


def test_signatures_match_contract():
    # The methods Dexa calls polymorphically must accept the same parameters as
    # the ABC declares, so call sites are backend-agnostic.
    for name in ("prefill", "reference_queries", "generate", "score"):
        base_sig = inspect.signature(getattr(ModelBackend, name))
        impl_sig = inspect.signature(getattr(VLLMBackend, name))
        assert list(base_sig.parameters) == list(impl_sig.parameters), name


def test_vllm_available_flag_is_bool():
    assert isinstance(vb.vllm_available(), bool)


@pytest.mark.skipif(vb.vllm_available(), reason="vllm IS installed; tested elsewhere")
def test_construction_without_vllm_raises_runtimeerror():
    with pytest.raises(RuntimeError) as ei:
        VLLMBackend(model_name="meta-llama/Llama-3.1-8B-Instruct")
    msg = str(ei.value).lower()
    assert "vllm" in msg  # helpful, mentions the missing dependency


@pytest.mark.skipif(vb.vllm_available(), reason="vllm IS installed; tested elsewhere")
def test_compact_decode_guard_without_vllm():
    # enable_compact_decode must refuse cleanly when vllm is unavailable.
    VLLMBackend._COMPACT_DECODE_READY = False
    with pytest.raises(RuntimeError):
        VLLMBackend.enable_compact_decode()


# --- functional: cluster-only, skipped without vllm ------------------------
@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("vllm")
    # Small model so the cluster smoke test is cheap; override as needed.
    return VLLMBackend(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        dtype="float32",
    )


def test_prefill_and_reference_shapes(backend):
    pytest.importorskip("vllm")
    s = backend.spec
    tokens = backend.tokenize("the quick brown fox jumps over the lazy dog")
    assert len(tokens) >= 2

    kv = backend.prefill(tokens)
    assert len(kv.layers) == s.n_layers
    for lk in kv.layers:
        assert lk.key.shape == (s.n_kv_heads, len(tokens), s.head_dim)
        assert lk.value.shape == (s.n_kv_heads, len(tokens), s.head_dim)
    assert kv.positions.shape == (len(tokens),)

    rq = backend.reference_queries(tokens, strategy="self", n_per_head=4)
    assert len(rq.layers) == s.n_layers
    for layer in rq.layers:
        assert layer.shape[0] == s.n_q_heads
        assert layer.shape[1] == min(4, len(tokens))
        assert layer.shape[2] == s.head_dim


def test_attention_outputs_model_free(backend):
    pytest.importorskip("vllm")
    s = backend.spec
    tokens = backend.tokenize("attention output reconstruction check")
    kv = backend.prefill(tokens)
    rq = backend.reference_queries(tokens, strategy="self")
    ao = backend.attention_outputs(kv, rq)
    assert len(ao) == s.n_layers
    assert ao[0].shape == (s.n_q_heads, len(tokens), s.head_dim)


def test_decode_requires_compact_shim(backend):
    pytest.importorskip("vllm")
    # Before enable_compact_decode(), generate/score must refuse rather than
    # silently return wrong numbers.
    VLLMBackend._COMPACT_DECODE_READY = False
    tokens = backend.tokenize("once upon a time")
    kv = backend.prefill(tokens)
    with pytest.raises(RuntimeError):
        backend.generate(kv, [], max_new_tokens=2)
    with pytest.raises(RuntimeError):
        backend.score(kv, [], tokens[:1])
