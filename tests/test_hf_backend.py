"""Correctness gate for the real HF model backend.

These prove the compaction *eval* path is faithful:
  (a) round-trip exactness of the KVCache + DynamicCache + position handling,
  (b) compact-decode exactness at the no-compression limit (keep-all, beta=0),
      which exercises the beta-injection / 4D-mask path,
  (c) plumbing sanity for generate() / score() shapes.

Run: ``.venv/bin/python -m pytest tests/test_hf_backend.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dexa.core.types import CompactCache, CompactLayer, KVCache
from dexa.engine.hf_backend import HFBackend

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def backend():
    return HFBackend(model_name=MODEL, device="cpu", dtype="float32")


def _keep_all_compact(kv: KVCache) -> CompactCache:
    """Build a CompactCache that keeps every key with beta=0 (identity)."""
    layers = []
    for lk in kv.layers:
        n_kv = lk.key.shape[0]
        layers.append(
            CompactLayer(
                keys=[lk.key[h].copy() for h in range(n_kv)],
                values=[lk.value[h].copy() for h in range(n_kv)],
                biases=[np.zeros(lk.key.shape[1], dtype=np.float32) for _ in range(n_kv)],
                positions=[kv.positions.copy().astype(np.int64) for _ in range(n_kv)],
            )
        )
    return CompactCache(
        spec=kv.spec,
        layers=layers,
        logical_length=kv.seq_len,
        method="keep_all",
        meta={"token_ids": list(kv.token_ids)},
    )


@pytest.mark.torch
def test_roundtrip_full_cache_matches_full_forward(backend):
    tokens = backend.tokenize("the quick brown fox jumps over the lazy dog today")
    assert len(tokens) >= 4
    ctx_tokens = tokens[:-3]
    next_tokens = tokens[-3:]

    # Reference: a single full forward over ctx+next, teacher-forced logprobs.
    full_ids = torch.tensor([ctx_tokens + next_tokens])
    with torch.no_grad():
        logits = backend.model(input_ids=full_ids).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    n_ctx = len(ctx_tokens)
    ref = np.array(
        [logprobs[n_ctx + k - 1, next_tokens[k]].item() for k in range(len(next_tokens))],
        dtype=np.float32,
    )

    cache = backend.prefill(ctx_tokens)
    got = backend.score(cache, [], next_tokens)

    assert got.shape == (len(next_tokens),)
    assert np.allclose(got, ref, atol=1e-3), f"max diff {np.abs(got - ref).max()}"


@pytest.mark.torch
def test_compact_identity_matches_full_cache(backend):
    tokens = backend.tokenize("alpha beta gamma delta epsilon zeta eta theta iota")
    ctx_tokens = tokens[:-2]
    next_tokens = tokens[-2:]

    cache = backend.prefill(ctx_tokens)
    compact = _keep_all_compact(cache)

    full_scores = backend.score(cache, [], next_tokens)
    compact_scores = backend.score(compact, [], next_tokens)

    assert np.allclose(compact_scores, full_scores, atol=1e-3), (
        f"max diff {np.abs(compact_scores - full_scores).max()}"
    )

    # Also exercise the path with a non-empty prompt.
    prompt = next_tokens[:1]
    target = next_tokens[1:]
    fp = backend.score(cache, prompt, target)
    cp = backend.score(compact, prompt, target)
    assert np.allclose(fp, cp, atol=1e-3)


@pytest.mark.torch
def test_generate_and_score_shapes(backend):
    tokens = backend.tokenize("once upon a time there was a small model")
    cache = backend.prefill(tokens)

    gen = backend.generate(cache, [], max_new_tokens=5)
    assert isinstance(gen, list) and len(gen) == 5
    assert all(isinstance(t, int) for t in gen)

    gen2 = backend.generate(cache, tokens[:2], max_new_tokens=3)
    assert len(gen2) == 3

    lp = backend.score(cache, [], tokens[:4])
    assert lp.shape == (4,)
    assert np.all(lp <= 0.0)


@pytest.mark.torch
def test_bf16_decode_no_overflow():
    """Regression: the mask/padded-bias sentinel must fit the active dtype.
    float32's min overflows bfloat16, which crashed score()/generate() on GPU
    (bf16) even though the float32 CPU tests passed."""
    from dexa.compaction.base import CompactionBudget
    from dexa.compaction.baselines import build

    be = HFBackend(model_name=MODEL, device="cpu", dtype="bfloat16")
    tokens = be.tokenize("the magic number for alpha is 4242 over the green hills today")
    full = be.prefill(tokens)
    refs = be.reference_queries(tokens, strategy="self", n_per_head=8)
    prompt = be.tokenize("what is the magic number")
    gold = be.tokenize("4242")

    lp_full = be.score(full, prompt, gold)               # the line that crashed
    assert np.all(np.isfinite(lp_full)) and lp_full.shape == (len(gold),)

    comp = build("attention_matching", value_ridge=0.05).compact(
        full, CompactionBudget(ratio=4.0), ref_queries=refs)
    lp_comp = be.score(comp, prompt, gold)               # exercises padded beta = sentinel
    assert np.all(np.isfinite(lp_comp))

    gen = be.generate(comp, prompt, max_new_tokens=3)
    assert len(gen) == 3


@pytest.mark.torch
def test_reference_queries_shapes(backend):
    tokens = backend.tokenize("reference query capture test sentence here now")
    s = backend.spec

    rq = backend.reference_queries(tokens, strategy="repeat_prefill", n_per_head=4)
    assert len(rq.layers) == s.n_layers
    for layer in rq.layers:
        assert layer.shape[0] == s.n_q_heads
        assert layer.shape[1] == min(4, len(tokens))
        assert layer.shape[2] == s.head_dim

    rq_self = backend.reference_queries(tokens, strategy="self")
    assert rq_self.layers[0].shape == (s.n_q_heads, len(tokens), s.head_dim)


@pytest.mark.torch
def test_attention_outputs_optional(backend):
    tokens = backend.tokenize("attention output reconstruction check")
    cache = backend.prefill(tokens)
    rq = backend.reference_queries(tokens, strategy="self")
    ao = backend.attention_outputs(cache, rq)
    s = backend.spec
    assert len(ao) == s.n_layers
    assert ao[0].shape == (s.n_q_heads, len(tokens), s.head_dim)
