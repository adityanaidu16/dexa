"""Benchmark metrics: quality (model-free + real) and system cost.

QUALITY
    * :func:`attention_recon_error` -- model-free signal that works with the
      FakeBackend. Compares the locally-normalized attention output of held-out
      queries over the full vs. compact cache (the very objective attention
      matching optimizes). Returns cosine similarity and relative L2 error.
    * :func:`answer_accuracy` -- real-model signal. Greedy-generates an answer
      against the (possibly compact) context and scores it against the gold.

SYSTEM (via :class:`~dexa.core.types.CostModel`)
    * :func:`system_metrics` -- KV bytes full vs. compact, memory saving,
      compression ratio, measured compaction wall time, modeled decode
      GPU-seconds (∝ attended tokens) and the prefill recompute it lets you
      avoid.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from dexa.core.types import CompactCache, CostModel, KVCache, RefQueries


def attention_recon_error(
    backend,
    full_cache: KVCache,
    compact_cache: CompactCache,
    eval_queries: RefQueries,
) -> dict[str, float]:
    """Reconstruction quality of ``compact_cache`` vs ``full_cache`` under
    ``eval_queries``. Mean over layers/heads/queries.

    Returns ``{cosine, rel_l2}`` where ``cosine`` is mean cosine similarity
    (1.0 == perfect) and ``rel_l2`` is mean ||full-comp|| / ||full|| (0 ==
    perfect). Returns NaNs if the backend can't compute attention outputs.
    """
    try:
        full = backend.attention_outputs(full_cache, eval_queries)
        comp = backend.attention_outputs(compact_cache, eval_queries)
    except NotImplementedError:
        return {"cosine": float("nan"), "rel_l2": float("nan")}

    cos_vals: list[float] = []
    l2_vals: list[float] = []
    for fo, co in zip(full, comp):
        # shapes [n_q_heads, n_ref, d] -> flatten heads*queries into rows
        a = fo.reshape(-1, fo.shape[-1])
        b = co.reshape(-1, co.shape[-1])
        an = np.linalg.norm(a, axis=-1)
        bn = np.linalg.norm(b, axis=-1)
        denom = np.clip(an * bn, 1e-8, None)
        cos = np.sum(a * b, axis=-1) / denom
        diff = np.linalg.norm(a - b, axis=-1)
        rel = diff / np.clip(an, 1e-8, None)
        cos_vals.append(float(np.mean(cos)))
        l2_vals.append(float(np.mean(rel)))
    return {"cosine": float(np.mean(cos_vals)), "rel_l2": float(np.mean(l2_vals))}


def answer_accuracy(backend, context_cache, task) -> dict[str, float]:
    """Greedy-generate an answer for ``task`` against ``context_cache`` and
    score it (exact-match + token-F1). Meaningful for a real LM; for the toy
    FakeBackend it just exercises the decode path."""
    gold = task.gold_ids
    max_new = max(4, len(gold) + 4)
    try:
        pred = backend.generate(
            context_cache, task.prompt_ids, max_new_tokens=max_new, greedy=True
        )
    except NotImplementedError:
        return {"exact_match": float("nan"), "token_f1": float("nan")}
    return task.scorer(list(pred), list(gold))


def system_metrics(
    cost: CostModel,
    full_cache: KVCache,
    compact_cache: Optional[CompactCache],
    compaction_seconds: float,
    *,
    n_decode: int = 64,
    decode_ref_len: float = 1000.0,
) -> dict[str, float]:
    """System cost summary for a (full -> compact) result.

    ``decode_gpu_seconds`` models the attention matmul cost: each decoded token
    attends over the whole cache, so cost ∝ cache length, normalized by
    ``decode_tok_per_s`` at a ``decode_ref_len`` reference context.
    """
    full_tokens = full_cache.seq_len

    if compact_cache is None:  # FullKV reference row
        ratio = 1.0
        compact_eff_tokens = float(full_tokens)
        kv_bytes_full = cost.kv_bytes(full_tokens)
        kv_bytes_compact = kv_bytes_full
        nbytes_full = full_cache.nbytes()
        nbytes_compact = nbytes_full
    else:
        ratio = float(compact_cache.compression_ratio)
        compact_eff_tokens = full_tokens / ratio if ratio else float(full_tokens)
        kv_bytes_full = cost.kv_bytes(full_tokens)
        kv_bytes_compact = cost.kv_bytes(compact_eff_tokens)
        nbytes_full = full_cache.nbytes()
        nbytes_compact = compact_cache.nbytes()

    def _decode_s(cache_len: float) -> float:
        return n_decode * cache_len / max(1e-9, cost.decode_tok_per_s * decode_ref_len)

    decode_full = _decode_s(full_tokens)
    decode_compact = _decode_s(compact_eff_tokens)

    return {
        "full_tokens": int(full_tokens),
        "compact_eff_tokens": float(compact_eff_tokens),
        "compression_ratio": float(ratio),
        "kv_bytes_full": float(kv_bytes_full),
        "kv_bytes_compact": float(kv_bytes_compact),
        "memory_saving": float(1.0 - kv_bytes_compact / max(1e-9, kv_bytes_full)),
        "nbytes_full": int(nbytes_full),
        "nbytes_compact": int(nbytes_compact),
        "nbytes_saving": float(1.0 - nbytes_compact / max(1e-9, nbytes_full)),
        "compaction_seconds": float(compaction_seconds),
        "decode_gpu_seconds_full": float(decode_full),
        "decode_gpu_seconds_compact": float(decode_compact),
        "decode_gpu_seconds_saved": float(decode_full - decode_compact),
        # reusing a stored compact cache avoids re-prefilling the context:
        "recompute_avoided_seconds": float(cost.prefill_seconds(full_tokens)),
    }


class _Timer:
    """``with _Timer() as t: ...`` then read ``t.seconds``."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False
