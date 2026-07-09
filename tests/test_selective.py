"""Pure selective-recompute logic (no model): deviation scoring, HKVD selection,
and KV blending."""

from __future__ import annotations

import numpy as np

from dexa.core.types import KVCache, LayerKV, ModelSpec
from dexa.segment.selective import (
    blend_kv,
    hkvd_select,
    per_token_kv_deviation,
)


def _kv(vals):
    """One-layer, one-head KVCache with key==value==vals[:, None] over T tokens."""
    vals = np.asarray(vals, dtype=np.float32)
    T = vals.shape[0]
    spec = ModelSpec(name="m", n_layers=1, n_q_heads=1, n_kv_heads=1, head_dim=2,
                     hidden_size=2)
    key = np.stack([vals, vals], axis=-1)[None]   # [1, T, 2]
    return KVCache(spec=spec, layers=[LayerKV(key=key.copy(), value=key.copy())],
                   positions=np.arange(T), token_ids=list(range(T)))


def test_deviation_ranks_changed_tokens():
    reused = _kv([0, 0, 0, 0, 0])
    correct = _kv([0, 5, 0, 9, 1])   # tokens 1,3,4 differ; 3 most, then 1, then 4
    dev = per_token_kv_deviation(reused, correct, (0, 5))
    order = list(np.argsort(dev)[::-1])
    assert order[:3] == [3, 1, 4]
    assert dev[0] == 0 and dev[2] == 0


def test_hkvd_selects_top_fraction():
    dev = np.array([0.1, 9.0, 0.2, 8.0, 0.0])
    sel = hkvd_select(dev, 0.4)          # ceil(0.4*5)=2 -> tokens 1 and 3
    assert list(sel) == [1, 3]
    assert list(hkvd_select(dev, 0.0)) == []
    assert list(hkvd_select(dev, 1.0)) == [0, 1, 2, 3, 4]


def test_hkvd_offset_and_baselines():
    dev = np.array([5.0, 0.0, 0.0, 5.0])
    assert list(hkvd_select(dev, 0.5, offset=10)) == [10, 13]      # top-2 + offset
    assert list(hkvd_select(dev, 0.5, strategy="recent")) == [2, 3]
    r1 = hkvd_select(dev, 0.5, strategy="random")
    r2 = hkvd_select(dev, 0.5, strategy="random")
    assert list(r1) == list(r2)          # deterministic


def test_blend_replaces_selected_reuses_rest():
    reused = _kv([0, 0, 0, 0])
    correct = _kv([1, 2, 3, 4])
    blended = blend_kv(reused, correct, np.array([1, 3]))
    got = blended.layers[0].key[0, :, 0]
    assert list(got) == [0, 2, 0, 4]     # 1,3 corrected; 0,2 reused
    # inputs untouched
    assert reused.layers[0].key[0, 1, 0] == 0


def test_deviation_layer_subset():
    reused = _kv([0, 0, 0])
    correct = _kv([0, 3, 0])
    full = per_token_kv_deviation(reused, correct, (0, 3), layers="all")
    one = per_token_kv_deviation(reused, correct, (0, 3), layers=1)
    assert np.array_equal(full, one)     # single-layer cache: identical
