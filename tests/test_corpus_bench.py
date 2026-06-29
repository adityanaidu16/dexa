"""Fast correctness gate for the GTM corpus-QA benchmark (dexa.bench.corpus).

Uses the tiny-random Llama with a tiny corpus and steps=2 so the whole module
runs in seconds. Proves the three-way comparison is wired correctly:
  * one row per method (full_context / rag / cartridge),
  * full_context is the quality ceiling (~1.0 by construction),
  * the cartridge path runs end-to-end and yields a valid Cartridge,
  * the TF-IDF retriever recovers the planted fact chunk for its question,
  * the break-even computation is present and sane.

Run: ``.venv/bin/python -m pytest tests/test_corpus_bench.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from dexa.core.types import CostModel
from dexa.engine.hf_backend import HFBackend
from dexa.bench.corpus import (
    BowRetriever,
    chunk_corpus,
    make_corpus_qa,
    report_corpus,
    run_corpus_bench,
)

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CART_OPTS = {"t": 8, "steps": 2, "n_selfstudy": 2, "answer_len": 4, "verbose": False}


@pytest.fixture(scope="module")
def backend():
    return HFBackend(model_name=MODEL, device="cpu", dtype="float32")


@pytest.fixture(scope="module")
def corpus_qa(backend):
    return make_corpus_qa(backend, n_facts=2, filler_tokens=48, seed=0)


# --- corpus + QA generation ------------------------------------------------
def test_make_corpus_qa_shapes(backend, corpus_qa):
    corpus, qa = corpus_qa
    assert isinstance(corpus, str) and len(corpus) > 0
    assert len(qa) == 2
    for q_ids, g_ids in qa:
        assert isinstance(q_ids, list) and len(q_ids) > 0
        assert isinstance(g_ids, list) and len(g_ids) > 0


# --- TF-IDF retriever recovers the planted fact ----------------------------
def test_rag_retriever_returns_planted_chunk(backend, corpus_qa):
    corpus, qa = corpus_qa
    corpus_ids = backend.tokenize(corpus)
    chunks = chunk_corpus(corpus_ids, chunk_tokens=16)
    assert len(chunks) >= 2
    retriever = BowRetriever(chunks)

    for q_ids, g_ids in qa:
        idxs = retriever.retrieve(q_ids, k=3)
        assert 1 <= len(idxs) <= 3
        # the meaningful, non-brittle check: the retriever recovered a chunk that
        # actually contains the fact's value tokens (the answer to this
        # question), not random filler. Token-id membership avoids detokenizer
        # round-trip fragility (a number detokenizes differently in isolation).
        gold_set = set(g_ids)
        fact_chunks = {i for i, c in enumerate(chunks) if gold_set & set(c)}
        assert fact_chunks, "planted gold not found in any chunk (fixture bug)"
        assert fact_chunks & set(idxs), f"retriever missed the fact chunk {fact_chunks}, got {idxs}"


# --- the full three-way benchmark ------------------------------------------
@pytest.mark.torch
def test_run_corpus_bench_end_to_end(backend, corpus_qa, tmp_path):
    corpus, qa = corpus_qa
    out = tmp_path / "corpus.json"
    results = run_corpus_bench(
        backend, corpus, qa,
        methods=["full_context", "rag", "cartridge"],
        cartridge_opts=CART_OPTS,
        rag_k=2, chunk_tokens=16,
        cost=CostModel(), n_decode=8,
        out_path=out, verbose=False,
    )

    rows = results["methods"]
    # a row per method.
    assert set(rows) == {"full_context", "rag", "cartridge"}
    for m, r in rows.items():
        for key in ("quality", "quality_exact_match", "memory_bytes",
                    "prefill_cost_s", "per_query_cost_s"):
            assert key in r and np.isfinite(r[key]), f"{m}.{key}"

    # full_context is the ceiling: recall fraction == 1.0 by construction.
    assert rows["full_context"]["quality"] == pytest.approx(1.0, abs=1e-6)

    # cartridge produced a valid artifact (uniform t, finite K/V, right shapes).
    cart = results["cartridge"]
    s = backend.spec
    assert cart is not None
    assert cart.t == 8
    assert cart.keys.shape == (s.n_layers, s.n_kv_heads, 8, s.head_dim)
    assert cart.values.shape == cart.keys.shape
    assert np.isfinite(cart.keys).all() and np.isfinite(cart.values).all()
    assert cart.logical_length == len(backend.tokenize(corpus))
    # it round-trips to a usable compact cache.
    cc = cart.to_compact_cache()
    assert cc.method == "cartridge" and len(cc.layers) == s.n_layers

    # memory ordering: cartridge resident KV is far below the full corpus KV.
    assert rows["cartridge"]["memory_bytes"] < rows["full_context"]["memory_bytes"]
    # per-query cost ordering: cartridge attends fewer tokens than full context.
    assert rows["cartridge"]["per_query_cost_s"] < rows["full_context"]["per_query_cost_s"]

    # break-even present and sane (non-negative ints, or None == "never").
    be = rows["cartridge"]["break_even"]
    assert "cartridge_vs_full_context" in be and "cartridge_vs_rag" in be
    for v in be.values():
        assert v is None or (isinstance(v, float) and v >= 0)
    # training costs more up-front than a single corpus prefill, so it does not
    # win full-context from query 0.
    assert be["cartridge_vs_full_context"] is None or be["cartridge_vs_full_context"] >= 1

    assert out.exists()


# --- reporting is guarded and does not raise -------------------------------
@pytest.mark.torch
def test_report_corpus_runs(backend, corpus_qa, tmp_path):
    corpus, qa = corpus_qa
    results = run_corpus_bench(
        backend, corpus, qa,
        cartridge_opts=CART_OPTS, rag_k=2, chunk_tokens=16,
        n_decode=8, out_path=None, verbose=False,
    )
    rep = report_corpus(results, out_dir=str(tmp_path))
    assert "table" in rep and "full_context" in rep["table"]
    assert "plots" in rep
