"""Attention Matching compaction.

Implements "Fast KV Compaction via Attention Matching" (Zweiger, Fu, Guo, Kim,
2026). The idea: replace a layer's full keys/values ``K, V`` [T, d] with a much
smaller ``Ck, Cv`` [t, d] plus per-key attention biases ``beta`` [t] chosen so
that, for a set of *reference queries*, the locally-normalized attention output

    softmax(q . Ck^T * scale + beta) . Cv

reproduces the full-cache output ``softmax(q . K^T * scale) . V`` as closely as
possible. ``scale = 1/sqrt(d)`` to match :class:`~dexa.engine.fake.FakeBackend`.

The fit is done independently per layer and per kv-head. Under grouped-query
attention the query heads sharing a kv-head are aggregated (their reference
queries are stacked) so the single compact head serves the whole group.

Three stages per kv-head:

1. **Key selection** -- which original keys to keep (``highest_attention`` RMS
   importance, or greedy ``omp`` on the mass-matching residual).
2. **Bias fit** -- non-negative least squares so the kept keys reproduce the
   total attention mass each reference query placed on the full key set;
   ``beta = log(w)``.
3. **Value fit** -- least squares mapping the compact attention weights back to
   the full attention outputs to obtain ``Cv``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import nnls

from dexa.compaction.base import CompactionBudget, Compactor
from dexa.core.types import (
    CompactCache,
    CompactLayer,
    KVCache,
    RefQueries,
)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax (subtract max before exp)."""
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


class AttentionMatching(Compactor):
    """Flagship compactor: fit compact keys/values/biases to reference queries.

    Parameters
    ----------
    key_selection:
        ``"highest_attention"`` (default) -- RMS attention importance top-t.
        ``"omp"`` -- greedy orthogonal matching pursuit on the mass residual.
    """

    name = "attention_matching"
    needs_ref_queries = True

    def __init__(self, key_selection: str = "highest_attention") -> None:
        if key_selection not in ("highest_attention", "omp"):
            raise ValueError(f"unknown key_selection {key_selection!r}")
        self.key_selection = key_selection

    # --- public API -------------------------------------------------------
    def compact(
        self,
        cache: KVCache,
        budget: CompactionBudget,
        *,
        ref_queries: Optional[RefQueries] = None,
    ) -> CompactCache:
        if ref_queries is None:
            raise ValueError("AttentionMatching requires ref_queries")

        spec = cache.spec
        scale = 1.0 / np.sqrt(spec.head_dim)
        T = cache.seq_len
        t = min(budget.target_t(T), T)
        positions = cache.positions

        layers: list[CompactLayer] = []
        for l in range(spec.n_layers):
            K_all = cache.layers[l].key       # [n_kv_heads, T, d]
            V_all = cache.layers[l].value     # [n_kv_heads, T, d]
            Q_all = ref_queries.layers[l]     # [n_q_heads, n_ref, d]

            keys: list[np.ndarray] = []
            values: list[np.ndarray] = []
            biases: list[np.ndarray] = []
            poss: list[np.ndarray] = []

            for h in range(spec.n_kv_heads):
                K = K_all[h].astype(np.float32)   # [T, d]
                V = V_all[h].astype(np.float32)   # [T, d]
                # Aggregate every q-head that maps to this kv-head (GQA).
                q_heads = [qh for qh in range(spec.n_q_heads) if spec.kv_head_of(qh) == h]
                Q = np.concatenate([Q_all[qh] for qh in q_heads], axis=0).astype(np.float32)
                # [n_ref_total, d]

                Ck, Cv, beta, S = self._fit_head(Q, K, V, t, scale)
                keys.append(Ck.astype(np.float32))
                values.append(Cv.astype(np.float32))
                biases.append(beta.astype(np.float32))
                poss.append(positions[S].astype(positions.dtype))

            layers.append(CompactLayer(keys=keys, values=values, biases=biases, positions=poss))

        return CompactCache(
            spec=spec,
            layers=layers,
            logical_length=T,
            method="attention_matching",
            meta={"key_selection": self.key_selection, "target_t": t},
        )

    # --- per-head fit -----------------------------------------------------
    def _fit_head(
        self,
        Q: np.ndarray,   # [n_ref, d]
        K: np.ndarray,   # [T, d]
        V: np.ndarray,   # [T, d]
        t: int,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (Ck [t,d], Cv [t,d], beta [t], S [t] indices)."""
        T = K.shape[0]
        # Trivial / no-op compaction: keep everything verbatim.
        if t >= T:
            S = np.arange(T)
            beta = np.zeros(T, dtype=np.float32)
            return K.copy(), V.copy(), beta, S

        full_logits = (Q @ K.T) * scale          # [n_ref, T]

        # STEP 1: key selection.
        if self.key_selection == "omp":
            S = self._select_omp(full_logits, t)
        else:
            S = self._select_highest_attention(full_logits)[:t]
        S = np.sort(S)

        Ck = K[S]                                  # [t, d]

        # STEP 2: bias fit (NNLS mass matching).
        beta = self._fit_bias(Q, Ck, full_logits, scale)

        # STEP 3: value fit (least squares).
        Cv = self._fit_values(Q, Ck, K, V, beta, scale)

        return Ck, Cv, beta, S

    # --- step 1 variants --------------------------------------------------
    @staticmethod
    def _select_highest_attention(full_logits: np.ndarray) -> np.ndarray:
        """Top keys by RMS attention probability across reference queries."""
        A = _softmax(full_logits, axis=-1)            # [n_ref, T]
        importance = np.sqrt(np.mean(A ** 2, axis=0))  # [T]
        order = np.argsort(importance)[::-1]
        return order

    @staticmethod
    def _select_omp(full_logits: np.ndarray, t: int) -> np.ndarray:
        """Greedy OMP on the mass-matching residual (Algorithm 1).

        Columns of ``Phi`` are exp(scaled logits) per full key; ``m`` is the
        per-query total mass. Greedily add the key whose column best correlates
        with the current residual, refit weights via NNLS, repeat.
        """
        # Per-row stabilization (constant per query cancels in the fit).
        Z = full_logits - full_logits.max(axis=1, keepdims=True)
        Phi = np.exp(Z)                       # [n_ref, T]
        m = Phi.sum(axis=1)                   # [n_ref]

        T = Phi.shape[1]
        selected: list[int] = []
        residual = m.copy()
        col_norms = np.linalg.norm(Phi, axis=0) + 1e-12

        for _ in range(min(t, T)):
            corr = np.abs(Phi.T @ residual) / col_norms   # [T]
            corr[selected] = -np.inf
            j = int(np.argmax(corr))
            selected.append(j)
            A = Phi[:, selected]
            w, _ = nnls(A, m)
            residual = m - A @ w

        return np.array(selected, dtype=int)

    # --- step 2 -----------------------------------------------------------
    @staticmethod
    def _fit_bias(
        Q: np.ndarray,        # [n_ref, d]
        Ck: np.ndarray,       # [t, d]
        full_logits: np.ndarray,  # [n_ref, T]
        scale: float,
    ) -> np.ndarray:
        """NNLS mass matching: solve min_{w>=0} ||A_mat w - m||, beta=log(w)."""
        compact_logits = (Q @ Ck.T) * scale       # [n_ref, t]
        # Stabilize: subtract per-query max over the *full* logits so A_mat and
        # m share a scale. The constant cancels exactly in the equation A w = m.
        c = full_logits.max(axis=1, keepdims=True)  # [n_ref, 1]
        A_mat = np.exp(compact_logits - c)          # [n_ref, t]
        m = np.exp(full_logits - c).sum(axis=1)     # [n_ref]

        w, _ = nnls(A_mat, m)
        beta = np.log(np.clip(w, 1e-9, None))
        return beta.astype(np.float32)

    # --- step 3 -----------------------------------------------------------
    @staticmethod
    def _fit_values(
        Q: np.ndarray,    # [n_ref, d]
        Ck: np.ndarray,   # [t, d]
        K: np.ndarray,    # [T, d]
        V: np.ndarray,    # [T, d]
        beta: np.ndarray,  # [t]
        scale: float,
    ) -> np.ndarray:
        """Least squares: Cv = argmin ||X Cv - Y||, X compact weights, Y full out."""
        X = _softmax((Q @ Ck.T) * scale + beta[None, :], axis=-1)  # [n_ref, t]
        Y = _softmax((Q @ K.T) * scale, axis=-1) @ V               # [n_ref, d]
        Cv, *_ = np.linalg.lstsq(X, Y, rcond=None)                 # [t, d]
        return Cv.astype(np.float32)
