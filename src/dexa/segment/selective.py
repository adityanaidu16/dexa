"""Selective KV recomputation — the CacheBlend (EuroSys'25) mechanism, applied to
Dexa's *mutation* case (README Phase 1, Layer C).

The gap this closes. Exact incremental recompute (`recompute_incremental`) reuses
the unchanged segment **prefix** and recomputes everything from the first edit
onward — the same reuse extent as prefix caching. But the tokens *after* a
mid-context edit usually have **identical content**; only their attention over the
edited region changed. Recomputing all of them is wasteful. CacheBlend's insight:
reuse their (stale) KV and recompute only the small fraction of tokens whose KV
**deviates most** from the correct value — the High-KV-Deviation (HKVD) tokens —
which restores most of the quality at a fraction of the compute.

This module is the **pure** selection logic (numpy/stdlib): given the reused
(stale) KV and the correct KV over a token range, score per-token deviation and
pick which tokens to recompute. The engine assembles the blended cache
(:meth:`dexa.engine.hf_backend.HFBackend.recompute_selective`).

Scope of v1. The clean, position-shift-free case: a **length-preserving** edit, so
downstream tokens keep their absolute positions (no RoPE re-phasing) and the only
staleness is cross-attention to the edited region — exactly the HKVD setting.
Length-changing edits additionally need exact RoPE re-phasing of the reused keys
(rotate by the position delta; keys are post-RoPE, RoPE composes) — the next
increment.
"""

from __future__ import annotations

import numpy as np

from dexa.core.types import KVCache


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """HF Llama ``rotate_half``: split the last dim in two, return (-x2, x1)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def rope_rephase_keys(keys: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Re-phase post-RoPE keys by a fixed position **delta** — exactly.

    A key cached at position ``p`` is ``R(p)·k_raw``; to reuse it at ``p+delta`` we
    need ``R(p+delta)·k_raw = R(delta)·R(p)·k_raw``, i.e. apply the rotation for
    ``delta`` to the cached key. ``cos``/``sin`` (shape ``[head_dim]``) encode
    ``R(delta)`` in HF's rotate-half convention (``cos(delta·inv_freq)`` duplicated
    across the two halves), so this is the exact same transform HF applies during a
    forward pass — ``k·cos + rotate_half(k)·sin`` — but for the *delta* angle. The
    result is bit-exact to prefilling those tokens at the shifted positions (up to
    fp), with no model forward. Values carry no RoPE and are reused unchanged.

    ``keys``: ``[n_kv_heads, T, head_dim]``; ``cos``/``sin``: ``[head_dim]``.
    """
    return keys * cos + _rotate_half(keys) * sin


def per_token_kv_deviation(
    reused: KVCache, correct: KVCache, token_range: tuple[int, int], *, layers: str | int = "all"
) -> np.ndarray:
    """L2 deviation between the reused (stale) and correct KV, per token, over
    ``token_range`` = [start, end).

    Summed over key+value, heads and head_dim, and over layers (``layers="all"``)
    or just the first ``layers`` layers — CacheBlend observes that the HKVD token
    *set* is largely stable across layers, so a cheap first-layers estimate ranks
    tokens almost as well as the full (oracle) deviation. Returns an array of
    length ``end - start`` aligned to ``token_range``.
    """
    start, end = token_range
    n_layers = len(reused.layers)
    use = n_layers if layers == "all" else min(int(layers), n_layers)
    dev = np.zeros(end - start, dtype=np.float64)
    for li in range(use):
        rk, ck = reused.layers[li], correct.layers[li]
        dk = ck.key[:, start:end] - rk.key[:, start:end]      # [n_kv, T, d]
        dv = ck.value[:, start:end] - rk.value[:, start:end]
        # L2 over heads+dim, per token.
        dev += np.sqrt((dk ** 2).sum(axis=(0, 2)) + (dv ** 2).sum(axis=(0, 2)))
    return dev


def hkvd_select(
    deviation: np.ndarray, frac: float, *, offset: int = 0, strategy: str = "hkvd"
) -> np.ndarray:
    """Choose which tokens (absolute indices = local index + ``offset``) to
    recompute.

    ``strategy``: ``"hkvd"`` picks the top-``frac`` by deviation (CacheBlend);
    ``"recent"`` picks the last ``frac`` of the range (a recency baseline);
    ``"random"`` is a deterministic pseudo-random baseline (seeded by deviation
    order so it is reproducible without global RNG state). Returns sorted absolute
    indices."""
    n = len(deviation)
    k = int(np.ceil(frac * n)) if frac > 0 else 0
    k = max(0, min(k, n))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if strategy == "hkvd":
        idx = np.argsort(deviation)[::-1][:k]
    elif strategy == "recent":
        idx = np.arange(n - k, n)
    elif strategy == "random":
        # deterministic shuffle: rank by a fixed hash of position, no global RNG.
        order = np.argsort(((np.arange(n) * 2654435761) & 0xFFFFFFFF))
        idx = order[:k]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return np.sort(idx.astype(np.int64)) + offset


def blend_kv(reused: KVCache, correct: KVCache, recompute_idx: np.ndarray) -> KVCache:
    """Build the blended KVCache: correct KV at ``recompute_idx``, reused elsewhere.

    (The reused cache is the base; the selected tokens' columns are overwritten with
    their correct KV.) Returns a new KVCache; inputs are not mutated."""
    sel = np.asarray(recompute_idx, dtype=np.int64)
    layers = []
    from dexa.core.types import LayerKV
    for rl, cl in zip(reused.layers, correct.layers):
        k = rl.key.copy()
        v = rl.value.copy()
        if sel.size:
            k[:, sel] = cl.key[:, sel]
            v[:, sel] = cl.value[:, sel]
        layers.append(LayerKV(key=k, value=v))
    return KVCache(spec=reused.spec, layers=layers, positions=reused.positions.copy(),
                   token_ids=list(correct.token_ids) if correct.token_ids is not None else None,
                   meta=dict(reused.meta))
