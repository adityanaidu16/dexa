"""The GTM corpus-QA benchmark: full-context vs RAG vs cartridge.

This is the launch artifact behind ``docs/CARTRIDGES.md``. It puts a single
static corpus "in context" three ways and measures the trade the product claims:

  1. **full_context** -- prefill the whole corpus once and score/answer every
     query against the full :class:`~dexa.core.types.KVCache`. The quality
     *ceiling*, but the entire corpus KV must stay resident (huge memory) and
     every decoded token re-attends over the full corpus length.
  2. **rag** -- split the corpus into chunks, retrieve the top-k by a
     dependency-light TF-IDF / bag-of-words cosine retriever (numpy only), and
     prefill *only* those chunks as the context for that query. Cheap and small,
     but lossy when retrieval misses the planted fact.
  3. **cartridge** -- compile a cartridge once with
     :class:`~dexa.cartridge.compiler.CartridgeCompiler`, then score/answer every
     query against ``cartridge.to_compact_cache()``. Tiny resident memory, decode
     attends only ``t`` compact tokens, and quality near the full-context ceiling
     -- at the price of one-time offline training.

The headline metrics, per method:
  * **quality** -- mean *needle-recall fraction* ``(lp - floor)/(ceiling - floor)``
    (full_context = ceiling = 1.0, no-context = floor = 0.0) plus exact-match
    from greedy generation.
  * **memory_bytes** -- resident KV / cartridge bytes (actual ``nbytes`` and a
    :class:`~dexa.core.types.CostModel` projection for a real model).
  * **prefill cost** -- one-time GPU-seconds (corpus prefill for full / a free
    TF-IDF index for rag / a training-cost proxy for the cartridge).
  * **per-query cost** -- GPU-seconds (full re-attends the corpus-length KV; rag
    re-prefills k chunks each query; cartridge attends ``t`` tokens).
  * **break-even** -- the query count after which the cartridge's amortized cost
    (training + per-query) beats full-context / rag.

Everything is numpy + the backend; matplotlib and rich are optional and guarded.
Validated on CPU with the tiny-random Llama; run for real on SmolLM2 / a GPU
model via ``benchmarks/cartridge_bench.py``.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dexa.core.types import CostModel

# Reuse the bench vocabulary / style so the corpus reads like the rest of the
# suite and the toy backend sees familiar token-frequency structure.
from dexa.bench.tasks import _FILLER_VOCAB, _KEYS, _filler, _splice, token_match

# Attributes asked about each entity. Distinct, retrievable nouns so a TF-IDF
# retriever has a real signal to match the question against the fact chunk.
_ATTRS = (
    "altitude", "password", "serial", "balance", "voltage",
    "frequency", "capacity", "pressure", "latitude", "checksum",
    "quota", "revision",
)

_DEFAULT_METHODS = ("full_context", "rag", "cartridge")


# --------------------------------------------------------------------------- #
# Corpus + QA generation
# --------------------------------------------------------------------------- #
def make_corpus_qa(
    backend,
    *,
    n_facts: int = 8,
    filler_tokens: int = 512,
    seed: int = 0,
) -> tuple[str, list[tuple[list[int], list[int]]]]:
    """Plant ``n_facts`` distinct retrievable facts in filler and ask each.

    Each fact reads ``"The <attr> of <entity> is <value> ."`` and its question
    ``"What is the <attr> of <entity> ?"`` with gold ``<value>`` (a 7-digit
    magic number, like the NIAH tasks). Facts are spliced at evenly spread depths
    so they sit throughout the corpus, not bunched at one end.

    Returns ``(corpus_text, qa)`` where ``qa`` is a list of
    ``(question_token_ids, gold_token_ids)`` -- one per planted fact.
    """
    rng = random.Random(hash(("corpus_qa", n_facts, filler_tokens, seed)) & 0xFFFFFFFF)
    n_facts = min(n_facts, len(_KEYS), len(_ATTRS))
    entities = rng.sample(_KEYS, k=n_facts)
    attrs = rng.sample(_ATTRS, k=n_facts)
    values = [str(rng.randint(1_000_000, 9_999_999)) for _ in range(n_facts)]

    inserts: list[tuple[float, str]] = []
    qa: list[tuple[list[int], list[int]]] = []
    for i in range(n_facts):
        ent, attr, val = entities[i], attrs[i], values[i]
        depth = (i + 0.5) / n_facts
        inserts.append((depth, f"The {attr} of {ent} is {val} ."))
        question = f"What is the {attr} of {ent} ?"
        qa.append((backend.tokenize(question), backend.tokenize(val)))

    words = _splice(_filler(filler_tokens, rng), inserts)
    corpus_text = " ".join(words)
    return corpus_text, qa


# --------------------------------------------------------------------------- #
# Dependency-light TF-IDF / bag-of-words retriever (numpy only)
# --------------------------------------------------------------------------- #
def chunk_corpus(corpus_ids: list[int], chunk_tokens: int) -> list[list[int]]:
    """Split a token-id sequence into contiguous chunks of ``chunk_tokens``."""
    chunk_tokens = max(1, int(chunk_tokens))
    return [corpus_ids[i : i + chunk_tokens] for i in range(0, len(corpus_ids), chunk_tokens)]


class BowRetriever:
    """TF-IDF bag-of-words cosine retriever over token-id chunks.

    No external deps: builds a vocabulary over the chunk token ids, weights each
    chunk by ``tf * idf``, L2-normalizes, and ranks chunks by cosine similarity
    to the (same-weighting) query vector. Deliberately simple -- this is the
    cheap-but-lossy baseline the cartridge is meant to beat on quality.
    """

    def __init__(self, chunks: list[list[int]]) -> None:
        self.chunks = chunks
        vocab: dict[int, int] = {}
        for ch in chunks:
            for tok in set(ch):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        V = len(vocab)
        n_chunks = len(chunks)

        # document frequency -> smoothed idf.
        df = np.zeros(V, dtype=np.float64)
        for ch in chunks:
            for tok in set(ch):
                df[vocab[tok]] += 1.0
        self.idf = np.log((1.0 + n_chunks) / (1.0 + df)) + 1.0

        # chunk tf-idf matrix [n_chunks, V], L2-normalized rows.
        mat = np.zeros((max(1, n_chunks), V), dtype=np.float64)
        for ci, ch in enumerate(chunks):
            for tok in ch:
                mat[ci, vocab[tok]] += 1.0
        mat *= self.idf[None, :]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._mat = mat / np.clip(norms, 1e-12, None)

    def _vec(self, token_ids: list[int]) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float64)
        for tok in token_ids:
            j = self.vocab.get(tok)
            if j is not None:
                v[j] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def scores(self, query_ids: list[int]) -> np.ndarray:
        """Cosine similarity of the query against every chunk."""
        if not self.chunks:
            return np.zeros(0, dtype=np.float64)
        return self._mat @ self._vec(query_ids)

    def retrieve(self, query_ids: list[int], k: int) -> list[int]:
        """Return the indices of the top-``k`` chunks (descending similarity)."""
        sims = self.scores(query_ids)
        if sims.size == 0:
            return []
        k = max(1, min(k, sims.size))
        # argsort descending, stable on ties for determinism.
        order = np.argsort(-sims, kind="stable")
        return [int(i) for i in order[:k]]


# --------------------------------------------------------------------------- #
# Cost model helpers
# --------------------------------------------------------------------------- #
def _decode_seconds(cost: CostModel, cache_len: float, n_decode: int, ref_len: float) -> float:
    """GPU-seconds to decode ``n_decode`` tokens, each attending ``cache_len``
    cached tokens (∝ cache length, normalized at ``ref_len``)."""
    return n_decode * cache_len / max(1e-9, cost.decode_tok_per_s * ref_len)


def _cartridge_train_proxy(
    cost: CostModel, *, T: int, t: int, steps: int, n_selfstudy: int, answer_len: int
) -> float:
    """A hardware-relative GPU-seconds proxy for compiling a cartridge.

    Two terms dominate the real method: (1) self-study, where the teacher prefills
    the *full* corpus once per synthetic question and generates an answer, and
    (2) training, where each step forwards every self-study query over the small
    compact cache. This is intentionally a multiple of a single corpus prefill,
    so the benchmark is honest that the cartridge pays a real one-time cost.
    """
    selfstudy = n_selfstudy * (cost.prefill_seconds(T) + _decode_seconds(cost, T, answer_len, ref_len=1000.0))
    train = steps * n_selfstudy * cost.prefill_seconds(t + answer_len)
    return float(selfstudy + train)


def _break_even(fixed_cart: float, pq_cart: float, fixed_other: float, pq_other: float) -> Optional[float]:
    """Smallest query count Q where ``fixed_cart + Q*pq_cart`` <= the other
    method's total cost. ``0`` if the cartridge wins immediately; ``None`` if it
    never catches up (its per-query cost is not lower)."""
    denom = pq_other - pq_cart
    if denom <= 0:
        # cartridge is not cheaper per query; only wins if already cheaper fixed.
        return 0.0 if fixed_cart <= fixed_other else None
    q = (fixed_cart - fixed_other) / denom
    return float(math.ceil(max(0.0, q)))


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def _sum_logprob(backend, context, prompt_ids: list[int], gold_ids: list[int]) -> float:
    return float(np.sum(backend.score(context, prompt_ids, gold_ids)))


def run_corpus_bench(
    backend,
    corpus: str,
    qa: list[tuple[list[int], list[int]]],
    *,
    methods: list[str] = list(_DEFAULT_METHODS),
    cartridge_opts: Optional[dict] = None,
    rag_k: int = 3,
    chunk_tokens: int = 128,
    cost: CostModel = CostModel(),
    n_decode: int = 32,
    out_path: Optional[str | Path] = os.path.join("benchmarks", "out", "corpus.json"),
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full-context vs RAG vs cartridge corpus-QA benchmark.

    Returns a dict with one row per method (quality, memory, costs, break-even),
    plus the compiled :class:`Cartridge` (under ``"cartridge"``) and the config.
    Writes a JSON-safe copy to ``out_path`` (set ``None`` to skip).
    """
    cartridge_opts = dict(cartridge_opts or {})
    methods = list(methods)

    corpus_ids = backend.tokenize(corpus)
    T = len(corpus_ids)
    if verbose:
        print(f"[corpus-bench] T={T} tok  {len(qa)} queries  methods={methods}", flush=True)

    # --- one-time full-context prefill: the quality ceiling + the floor ------
    full = backend.prefill(corpus_ids)
    floor_cache = backend.prefill(corpus_ids[:1])  # no-context floor (prior alone)

    ceilings: list[float] = []
    floors: list[float] = []
    for q_ids, g_ids in qa:
        ceilings.append(_sum_logprob(backend, full, q_ids, g_ids))
        floors.append(_sum_logprob(backend, floor_cache, q_ids, g_ids))

    def _recall(lp: float, i: int) -> float:
        rng = ceilings[i] - floors[i]
        return (lp - floors[i]) / (rng if abs(rng) > 1e-6 else 1e-6)

    # --- RAG index (free, CPU) -----------------------------------------------
    chunks = chunk_corpus(corpus_ids, chunk_tokens)
    retriever = BowRetriever(chunks)

    # --- cartridge: compile once (timed) -------------------------------------
    cartridge = None
    compact = None
    compile_seconds = 0.0
    if "cartridge" in methods:
        from dexa.cartridge.compiler import CartridgeCompiler

        if verbose:
            print("[corpus-bench] compiling cartridge ...", flush=True)
        t0 = time.perf_counter()
        cartridge = CartridgeCompiler(backend).compile(corpus, **cartridge_opts)
        compile_seconds = time.perf_counter() - t0
        compact = cartridge.to_compact_cache()

    # --- evaluate every method ------------------------------------------------
    rows: dict[str, dict[str, Any]] = {}
    rag_ctx_tokens_total = 0
    rag_ctx_bytes = 0.0

    for method in methods:
        recalls: list[float] = []
        exacts: list[float] = []

        for i, (q_ids, g_ids) in enumerate(qa):
            if method == "full_context":
                ctx = full
            elif method == "rag":
                idxs = sorted(retriever.retrieve(q_ids, rag_k))  # original order
                retrieved = [tok for ci in idxs for tok in chunks[ci]]
                ctx = backend.prefill(retrieved) if retrieved else floor_cache
                rag_ctx_tokens_total += len(retrieved)
                if i == 0:
                    rag_ctx_bytes = float(ctx.nbytes())
            elif method == "cartridge":
                ctx = compact
            else:
                raise ValueError(f"unknown method: {method!r}")

            lp = _sum_logprob(backend, ctx, q_ids, g_ids)
            recalls.append(_recall(lp, i))

            max_new = max(4, len(g_ids) + 4)
            pred = backend.generate(ctx, q_ids, max_new_tokens=max_new, greedy=True)
            exacts.append(token_match(list(pred), list(g_ids))["exact_match"])

        # --- memory (resident KV / cartridge bytes) --------------------------
        if method == "full_context":
            mem_actual = float(full.nbytes())
            mem_model = float(cost.kv_bytes(T))
            ctx_len = float(T)
        elif method == "rag":
            mean_rag_tokens = rag_ctx_tokens_total / max(1, len(qa))
            mem_actual = rag_ctx_bytes  # only k chunks resident per query
            mem_model = float(cost.kv_bytes(mean_rag_tokens))
            ctx_len = float(mean_rag_tokens)
        else:  # cartridge
            mem_actual = float(cartridge.nbytes())
            mem_model = float(cost.kv_bytes(cartridge.t))
            ctx_len = float(cartridge.t)

        # --- costs (GPU-seconds) ---------------------------------------------
        if method == "full_context":
            prefill_cost = cost.prefill_seconds(T)            # one-time corpus prefill (prompt cache)
            per_query = _decode_seconds(cost, T, n_decode, ref_len=1000.0)
        elif method == "rag":
            prefill_cost = 0.0                                # TF-IDF index: CPU, negligible
            per_query = cost.prefill_seconds(ctx_len) + _decode_seconds(cost, ctx_len, n_decode, ref_len=1000.0)
        else:  # cartridge
            t = cartridge.t
            steps = int(cartridge_opts.get("steps", 120))
            n_ss = int(cartridge.meta.get("n_selfstudy", cartridge_opts.get("n_selfstudy", 6)))
            ans = int(cartridge_opts.get("answer_len", 24))
            prefill_cost = _cartridge_train_proxy(
                cost, T=T, t=t, steps=steps, n_selfstudy=n_ss, answer_len=ans
            )
            per_query = _decode_seconds(cost, t, n_decode, ref_len=1000.0)

        rows[method] = {
            "quality": float(np.mean(recalls)) if recalls else float("nan"),
            "quality_exact_match": float(np.mean(exacts)) if exacts else float("nan"),
            "memory_bytes": mem_actual,
            "memory_bytes_model": mem_model,
            "context_tokens": ctx_len,
            "prefill_cost_s": float(prefill_cost),
            "per_query_cost_s": float(per_query),
            "n_queries": len(qa),
        }

    # --- break-even: where the cartridge amortizes past full / rag -----------
    break_even: dict[str, Optional[float]] = {}
    if "cartridge" in rows:
        c = rows["cartridge"]
        for other in ("full_context", "rag"):
            if other in rows:
                o = rows[other]
                break_even[f"cartridge_vs_{other}"] = _break_even(
                    c["prefill_cost_s"], c["per_query_cost_s"],
                    o["prefill_cost_s"], o["per_query_cost_s"],
                )
        rows["cartridge"]["break_even"] = break_even

    results: dict[str, Any] = {
        "config": {
            "model": backend.spec.name,
            "T": T,
            "n_queries": len(qa),
            "methods": methods,
            "rag_k": rag_k,
            "chunk_tokens": chunk_tokens,
            "n_chunks": len(chunks),
            "n_decode": n_decode,
            "cartridge_opts": cartridge_opts,
            "cost_model": cost.name,
            "compile_seconds": compile_seconds,
        },
        "methods": rows,
        "break_even": break_even,
        "cartridge": cartridge,  # in-memory only; stripped from JSON
    }

    if out_path is not None:
        _save_json(results, out_path)
        if verbose:
            print(f"[corpus-bench] saved -> {out_path}", flush=True)
    return results


def _save_json(results: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: v for k, v in results.items() if k != "cartridge"}
    cart = results.get("cartridge")
    if cart is not None:
        safe["cartridge_meta"] = {
            "t": cart.t, "logical_length": cart.logical_length,
            "compression_ratio": cart.compression_ratio,
            "nbytes": cart.nbytes(), "meta": cart.meta,
        }
    out_path.write_text(json.dumps(safe, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Reporting: the docs/CARTRIDGES.md-style table + money plots
# --------------------------------------------------------------------------- #
def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _fmt_be(be: dict, method: str) -> str:
    """Break-even cell for a method row."""
    if method != "cartridge" or not be:
        return "-"
    parts = []
    for other in ("full_context", "rag"):
        v = be.get(f"cartridge_vs_{other}")
        tag = "full" if other == "full_context" else "rag"
        if v is None:
            parts.append(f"{tag}:never")
        else:
            parts.append(f"{tag}:{int(v)}q")
    return " ".join(parts)


def report_corpus(results: dict, out_dir: str = os.path.join("benchmarks", "out")) -> dict:
    """Print the docs-style table and (guarded) save the money plots.

    Table columns: method | quality | exact | memory | cost/query | break-even.
    Plots: quality-vs-memory and cumulative-cost-vs-#queries (the break-even
    chart). Returns ``{"table": str, "plots": {...}}``.
    """
    rows = results["methods"]
    be = results.get("break_even", {})
    order = [m for m in ("full_context", "rag", "cartridge") if m in rows]
    order += [m for m in rows if m not in order]

    headers = ["method", "quality", "exact", "memory", "cost/query (s)", "break-even"]
    cells = []
    for m in order:
        r = rows[m]
        cells.append([
            m,
            f"{r['quality']:.3f}",
            f"{r['quality_exact_match']:.2f}",
            _fmt_bytes(r["memory_bytes"]),
            f"{r['per_query_cost_s']:.2e}",
            _fmt_be(be, m),
        ])

    table = _render_table(headers, cells, title="Dexa cartridge GTM benchmark (corpus-QA)")
    print(table)

    plots = {
        "quality_vs_memory": _plot_quality_memory(results, out_dir),
        "break_even": _plot_break_even(results, out_dir),
    }
    for name, p in plots.items():
        if p:
            print(f"saved {name} plot -> {p}")
    if not any(plots.values()):
        print("(matplotlib not available; skipped plots)")
    return {"table": table, "plots": plots}


def _render_table(headers: list[str], cells: list[list[str]], *, title: str) -> str:
    try:
        from rich.console import Console
        from rich.table import Table
        import io

        table = Table(title=title)
        for h in headers:
            table.add_column(h, justify="right", no_wrap=True)
        for c in cells:
            table.add_row(*c)
        console = Console(record=True, width=120, file=io.StringIO())
        console.print(table)
        return console.export_text()
    except Exception:
        widths = [len(h) for h in headers]
        for c in cells:
            widths = [max(w, len(x)) for w, x in zip(widths, c)]
        lines = [title]
        lines.append("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
        lines.append("  ".join("-" * w for w in widths))
        for c in cells:
            lines.append("  ".join(x.rjust(w) for x, w in zip(c, widths)))
        return "\n".join(lines)


def _plot_quality_memory(results: dict, out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = results["methods"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, r in rows.items():
        ax.scatter(r["memory_bytes_model"], r["quality"], s=120, label=m)
        ax.annotate(m, (r["memory_bytes_model"], r["quality"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xscale("log")
    ax.set_xlabel("resident memory (modeled bytes, log scale)  -> smaller is better")
    ax.set_ylabel("quality (needle-recall fraction)  -> higher is better")
    ax.set_title("Quality vs memory: cartridge holds ceiling quality at tiny memory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "corpus_quality_vs_memory.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_break_even(results: dict, out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
    except Exception:
        return None

    rows = results["methods"]
    be = results.get("break_even", {})
    # x-axis: number of queries up to a bit past the largest finite break-even.
    bes = [v for v in be.values() if v is not None]
    q_max = max(50, int(max(bes) * 1.5) if bes else 50)
    xs = _np.arange(0, q_max + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m, r in rows.items():
        ys = r["prefill_cost_s"] + xs * r["per_query_cost_s"]
        ax.plot(xs, ys, label=m, linewidth=2)
    for other, label in (("full_context", "vs full"), ("rag", "vs rag")):
        v = be.get(f"cartridge_vs_{other}")
        if v is not None and 0 < v <= q_max:
            ax.axvline(v, color="gray", linestyle="--", alpha=0.6)
            ax.annotate(f"break-even {label}\n@ {int(v)}q", (v, 0),
                        textcoords="offset points", xytext=(4, 10), fontsize=8)
    ax.set_xlabel("number of queries over the corpus lifetime")
    ax.set_ylabel("cumulative GPU-seconds (one-time + per-query)")
    ax.set_title("Break-even: cartridge training amortizes over reuse")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "corpus_break_even.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
