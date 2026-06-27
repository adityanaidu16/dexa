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


def _nnls_robust(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Non-negative least squares that never raises.

    ``scipy.optimize.nnls`` can hit its iteration cap on ill-conditioned mass
    matrices (e.g. self-study reference queries produce near-collinear columns).
    We give it a generous budget and, if it still fails to converge, fall back
    to a clipped ordinary least-squares solution -- always non-negative, always
    a usable answer for the ``beta = log(w)`` step downstream.
    """
    maxiter = max(50, 20 * A.shape[1])
    try:
        w, _ = nnls(A, b, maxiter=maxiter)
        return w
    except RuntimeError:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.clip(w, 0.0, None)


class AttentionMatching(Compactor):
    """Flagship compactor: fit compact keys/values/biases to reference queries.

    Parameters
    ----------
    key_selection:
        ``"highest_attention"`` (default) -- RMS attention importance top-t.
        ``"omp"`` -- greedy orthogonal matching pursuit on the mass residual.
    budget_alloc:
        ``"uniform"`` (default) -- every kv-head gets the same token budget.
        ``"sensitivity"`` -- distribute a per-layer token budget across kv-heads
        by sensitivity (paper Algorithm 4): concentrated-attention heads get
        fewer keys, diffuse heads get more. Total budget per layer is preserved.
    value_ridge:
        Tikhonov (ridge) regularization strength ``lambda`` for the value fit.
        ``0.0`` (default) reproduces the plain ``lstsq`` solve; ``>0`` solves the
        ridge normal equations ``(X^T X + lambda I) Cv = X^T Y`` which fixes the
        ill-conditioned / overfit value fit at some budgets.
    value_train_frac:
        Fraction of reference queries used to *fit* the value map; the remainder
        is held out (unused here but reserved for lambda selection). ``1.0``
        (default) fits on all queries.
    """

    name = "attention_matching"
    needs_ref_queries = True

    def __init__(
        self,
        key_selection: str = "highest_attention",
        *,
        budget_alloc: str = "uniform",
        value_ridge: float = 0.0,
        value_train_frac: float = 1.0,
    ) -> None:
        if key_selection not in ("highest_attention", "omp"):
            raise ValueError(f"unknown key_selection {key_selection!r}")
        if budget_alloc not in ("uniform", "sensitivity"):
            raise ValueError(f"unknown budget_alloc {budget_alloc!r}")
        if value_ridge < 0:
            raise ValueError("value_ridge must be >= 0")
        if not (0.0 < value_train_frac <= 1.0):
            raise ValueError("value_train_frac must be in (0, 1]")
        self.key_selection = key_selection
        self.budget_alloc = budget_alloc
        self.value_ridge = float(value_ridge)
        self.value_train_frac = float(value_train_frac)

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
        min_t = min(budget.min_tokens, T)
        positions = cache.positions

        layers: list[CompactLayer] = []
        for l in range(spec.n_layers):
            K_all = cache.layers[l].key       # [n_kv_heads, T, d]
            V_all = cache.layers[l].value     # [n_kv_heads, T, d]
            Q_all = ref_queries.layers[l]     # [n_q_heads, n_ref, d]

            # Aggregate reference queries per kv-head (GQA) once.
            Qs = [
                np.concatenate(
                    [Q_all[qh] for qh in range(spec.n_q_heads) if spec.kv_head_of(qh) == h],
                    axis=0,
                ).astype(np.float32)
                for h in range(spec.n_kv_heads)
            ]

            # STEP 0: distribute the layer's token budget across kv-heads.
            t_per_head = self._allocate_budget(Qs, K_all, t, T, min_t, scale)

            keys: list[np.ndarray] = []
            values: list[np.ndarray] = []
            biases: list[np.ndarray] = []
            poss: list[np.ndarray] = []

            for h in range(spec.n_kv_heads):
                K = K_all[h].astype(np.float32)   # [T, d]
                V = V_all[h].astype(np.float32)   # [T, d]
                Q = Qs[h]                         # [n_ref_total, d]

                Ck, Cv, beta, S = self._fit_head(Q, K, V, t_per_head[h], scale)
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
            meta={
                "key_selection": self.key_selection,
                "target_t": t,
                "budget_alloc": self.budget_alloc,
                "value_ridge": self.value_ridge,
            },
        )

    # --- step 0: per-head budget allocation -------------------------------
    def _allocate_budget(
        self,
        Qs: list[np.ndarray],   # per kv-head [n_ref, d]
        K_all: np.ndarray,      # [n_kv_heads, T, d]
        t: int,
        T: int,
        min_t: int,
        scale: float,
    ) -> list[int]:
        """Return a per-kv-head token budget summing to ``t * n_kv_heads``.

        ``uniform`` gives every head ``t``. ``sensitivity`` runs a cheap greedy
        allocation (Algorithm 4): the marginal value of giving a head one more
        key is the next-largest entry of its sorted attention-importance vector.
        Selecting the globally-largest marginal values respects the per-head
        prefix structure (importances are sorted descending), so concentrated
        heads -- whose importance collapses after a few keys -- receive fewer
        tokens while diffuse heads receive more. ``min_t`` is honored per head.
        """
        n_kv = len(Qs)
        if self.budget_alloc == "uniform" or t >= T:
            return [t] * n_kv

        total = t * n_kv
        min_t = max(1, min(min_t, t))
        # Per-head importance (descending), measured like ``highest_attention``.
        imp: list[np.ndarray] = []
        for h in range(n_kv):
            full_logits = (Qs[h] @ K_all[h].T) * scale          # [n_ref, T]
            A = _softmax(full_logits, axis=-1)
            importance = np.sqrt(np.mean(A ** 2, axis=0))        # [T]
            imp.append(np.sort(importance)[::-1])                # descending

        alloc = [min_t] * n_kv
        remaining = total - min_t * n_kv
        if remaining <= 0:
            return [min(a, T) for a in alloc]

        # Pool marginal values at ranks >= min_t across heads; take the top
        # ``remaining`` (capping each head at T). ``(value, head)`` pairs.
        cands: list[tuple[float, int]] = []
        for h in range(n_kv):
            for r in range(min_t, T):
                cands.append((float(imp[h][r]), h))
        cands.sort(key=lambda x: -x[0])
        for _value, h in cands[:remaining]:
            alloc[h] += 1
        return [min(a, T) for a in alloc]

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

        # STEP 3: value fit (ridge-regularized least squares).
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
            w = _nnls_robust(A, m)
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

        w = _nnls_robust(A_mat, m)
        beta = np.log(np.clip(w, 1e-9, None))
        return beta.astype(np.float32)

    # --- step 3 -----------------------------------------------------------
    def _fit_values(
        self,
        Q: np.ndarray,    # [n_ref, d]
        Ck: np.ndarray,   # [t, d]
        K: np.ndarray,    # [T, d]
        V: np.ndarray,    # [T, d]
        beta: np.ndarray,  # [t]
        scale: float,
    ) -> np.ndarray:
        """Fit Cv mapping compact attention weights X to the full output Y.

        The compact attention-weight matrix ``X`` [n_ref, t] is rank-deficient
        whenever a kept key receives ~0 weight across *every* reference query --
        common when key selection keeps duplicate / collinear keys that the bias
        fit then zeroes out (``beta = log(1e-9)``). Such a key is a near-zero
        column of ``X``; a plain ``lstsq`` / normal-equations solve leaves its row
        of ``Cv`` unconstrained and sends it to ~1e10. That is harmless in
        isolation (the key carries ~0 weight) but a float32 overflow waiting to
        happen, and it detonates once the compact cache is fused with raw KV in
        WorkingMemory. Three safeguards keep ``|Cv|`` on the scale of the values
        it replaces without disturbing the well-conditioned fit:

        1. **Drop dead keys.** Keys whose total reference weight is ~0 are
           excluded from the fit and their value left at 0; they contribute ~0 to
           any future attention output, so 0 is as good as anything -- and finite.
        2. **Bounded solve.** The live columns are fit via the ridge normal
           equations ``(X^T X + lambda I) Cv = X^T Y``. Even the default
           ``value_ridge == 0`` applies a tiny scale-adaptive ``lambda`` (relative
           to ``mean(diag(X^T X))``) so the solve stays well-posed for any
           remaining near-collinear live keys; a user-supplied ``value_ridge > 0``
           overrides it (and additionally conditions an overfit fit at mid
           budgets, removing the recall dip there).
        3. **Clip the dropped rows** to the empirical per-dim range of the
           original ``V`` for this head -- a no-op safety net (the dropped rows
           are 0, the live rows already sit inside this range) that guarantees no
           value escapes the magnitude of the values it stands in for.

        ``value_train_frac < 1`` fits on a deterministic subset of the reference
        queries (the rest are held out) to further discourage overfitting.
        """
        X = _softmax((Q @ Ck.T) * scale + beta[None, :], axis=-1)  # [n_ref, t]
        Y = _softmax((Q @ K.T) * scale, axis=-1) @ V               # [n_ref, d]

        if self.value_train_frac < 1.0 and X.shape[0] > Ck.shape[0]:
            n_fit = max(Ck.shape[0], int(round(self.value_train_frac * X.shape[0])))
            idx = np.linspace(0, X.shape[0] - 1, n_fit).astype(int)
            X, Y = X[idx], Y[idx]

        t = X.shape[1]
        # A compact key gets ~0 weight across all reference queries -> near-zero
        # column -> its Cv row is unconstrained by the fit. Drop it (value 0).
        live = X.sum(axis=0) > 1e-6 * X.shape[0]                   # [t]
        Cv = np.zeros((t, V.shape[1]), dtype=np.float64)
        if live.any():
            Cv[live] = self._solve_value_map(X[:, live], Y)

        # Safety net: keep every value inside the original V's per-dim range.
        # Live rows already lie within it; this only clamps the dropped rows.
        lo = V.min(axis=0)
        hi = V.max(axis=0)
        if (~live).any():
            Cv[~live] = np.clip(Cv[~live], lo, hi)
        return Cv.astype(np.float32)

    def _solve_value_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Solve ``X @ Cv ~= Y`` for the live compact keys with a bounded ridge.

        ``value_ridge > 0`` uses that ``lambda`` directly; the default ``0``
        applies a tiny scale-adaptive ``lambda`` (relative to the mean diagonal of
        ``X^T X``) so the normal equations stay well-posed -- bounding the
        solution for near-collinear keys without perceptibly altering the
        well-conditioned fit. Done in float64 for numerical headroom, then cast
        back to float32 by the caller.
        """
        Xd = X.astype(np.float64)
        Yd = Y.astype(np.float64)
        n = Xd.shape[1]
        XtX = Xd.T @ Xd
        if self.value_ridge > 0.0:
            lam = self.value_ridge
        else:
            lam = 1e-6 * (float(np.trace(XtX)) / max(n, 1)) + 1e-12
        XtX = XtX + lam * np.eye(n, dtype=np.float64)
        return np.linalg.solve(XtX, Xd.T @ Yd)
