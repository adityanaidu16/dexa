"""A torch-free, deterministic backend for tests and CI.

It is a *real* attention mechanism over small numpy vectors — not a language
model. Its purpose is to exercise the compaction math and the benchmark plumbing
without a GPU. The honest quality signal it supports is **attention-output
reconstruction error** (the same objective Attention Matching optimizes):
``softmax(Q·Ck^T + beta)·Cv`` vs ``softmax(Q·K^T)·V`` over held-out queries.

Token "semantics": a token id deterministically maps to a per-head key/value/
query vector. Identical token ids share vectors, so a planted "needle" token has
a distinctive, recoverable KV — letting compaction quality move a measurable
metric even in the toy.
"""

from __future__ import annotations

import numpy as np

from dexa.core.types import (
    CompactCache,
    KVCache,
    LayerKV,
    ModelSpec,
    RefQueries,
)
from dexa.engine.base import ContextCache, ModelBackend


def _rng(*seed_parts: int) -> np.random.Generator:
    h = abs(hash(seed_parts)) % (2**32)
    return np.random.default_rng(h)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-8, None)


class FakeBackend(ModelBackend):
    def __init__(
        self,
        n_layers: int = 3,
        n_q_heads: int = 4,
        n_kv_heads: int = 2,
        head_dim: int = 8,
        seed: int = 0,
    ) -> None:
        self._spec = ModelSpec(
            name="fake-attn",
            n_layers=n_layers,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=n_q_heads * head_dim,
        )
        self._seed = seed

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    # --- tokenization: whitespace + hashing, deterministic ----------------
    def tokenize(self, text: str) -> list[int]:
        return [abs(hash((self._seed, w))) % 50000 for w in text.split()]

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(f"<{t}>" for t in token_ids)

    # --- vector builders --------------------------------------------------
    def _kv_vec(self, token_id: int, layer: int, kv_head: int, which: str) -> np.ndarray:
        return _unit(
            _rng(self._seed, token_id, layer, kv_head, hash(which)).standard_normal(
                self._spec.head_dim
            )
        )

    def _q_vec(self, token_id: int, layer: int, q_head: int) -> np.ndarray:
        # Query for a token aligns with the key of the *same* token id in the
        # same kv-group, so attention is retrieval-like and content-driven.
        kv_head = self._spec.kv_head_of(q_head)
        base = self._kv_vec(token_id, layer, kv_head, "k")
        noise = 0.15 * _rng(self._seed, token_id, layer, q_head, 7).standard_normal(
            self._spec.head_dim
        )
        return _unit(base + noise)

    # --- prefill ----------------------------------------------------------
    def prefill(self, token_ids: list[int], *, position_offset: int = 0) -> KVCache:
        T = len(token_ids)
        s = self._spec
        layers: list[LayerKV] = []
        for l in range(s.n_layers):
            K = np.zeros((s.n_kv_heads, T, s.head_dim), dtype=np.float32)
            V = np.zeros((s.n_kv_heads, T, s.head_dim), dtype=np.float32)
            for h in range(s.n_kv_heads):
                for i, tid in enumerate(token_ids):
                    K[h, i] = self._kv_vec(tid, l, h, "k")
                    V[h, i] = self._kv_vec(tid, l, h, "v")
            layers.append(LayerKV(key=K, value=V))
        positions = np.arange(position_offset, position_offset + T, dtype=np.int64)
        return KVCache(spec=s, layers=layers, positions=positions, token_ids=list(token_ids))

    def reference_queries(
        self, token_ids: list[int], *, strategy: str = "repeat_prefill", n_per_head: int = 512
    ) -> RefQueries:
        T = len(token_ids)
        s = self._spec
        layers: list[np.ndarray] = []
        for l in range(s.n_layers):
            Q = np.zeros((s.n_q_heads, T, s.head_dim), dtype=np.float32)
            for qh in range(s.n_q_heads):
                for i, tid in enumerate(token_ids):
                    Q[qh, i] = self._q_vec(tid, l, qh)
            if T > n_per_head:
                idx = np.linspace(0, T - 1, n_per_head).astype(int)
                Q = Q[:, idx]
            layers.append(Q)
        return RefQueries(spec=s, layers=layers)

    # --- attention outputs (the meaningful, model-free quality signal) ----
    def attention_outputs(self, cache: ContextCache, queries: RefQueries) -> list[np.ndarray]:
        s = self._spec
        scale = 1.0 / np.sqrt(s.head_dim)
        out: list[np.ndarray] = []
        for l in range(s.n_layers):
            Q = queries.layers[l]  # [n_q_heads, n_ref, d]
            n_ref = Q.shape[1]
            res = np.zeros((s.n_q_heads, n_ref, s.head_dim), dtype=np.float32)
            for qh in range(s.n_q_heads):
                h = s.kv_head_of(qh)
                if isinstance(cache, KVCache):
                    K = cache.layers[l].key[h]      # [T, d]
                    Vv = cache.layers[l].value[h]   # [T, d]
                    beta = None
                else:
                    K = cache.layers[l].keys[h]     # [t, d]
                    Vv = cache.layers[l].values[h]  # [t, d]
                    beta = cache.layers[l].biases[h]  # [t]
                logits = (Q[qh] @ K.T) * scale       # [n_ref, t]
                if beta is not None:
                    logits = logits + beta[None, :]
                logits -= logits.max(axis=-1, keepdims=True)
                w = np.exp(logits)
                w /= np.clip(w.sum(axis=-1, keepdims=True), 1e-8, None)
                res[qh] = w @ Vv
            out.append(res)
        return out

    # --- decode / scoring: trivial deterministic stubs (plumbing only) ----
    def generate(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 64,
        greedy: bool = True,
    ) -> list[int]:
        # Deterministic pseudo-decode from the last-layer attention output so
        # bench plumbing runs; not a language model.
        q = self.reference_queries(prompt_token_ids or [0])
        ao = self.attention_outputs(context, q)[-1]  # [n_q_heads, n_ref, d]
        vec = ao[:, -1, :].ravel()
        toks = []
        for k in range(max_new_tokens):
            toks.append(int(abs(hash((self._seed, "gen", k, *np.round(vec, 3)))) % 50000))
        return toks

    def score(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
    ) -> np.ndarray:
        # Uniform-ish deterministic logprob; plumbing only.
        return np.full(len(target_token_ids), -np.log(50000.0), dtype=np.float32)
