"""STILL compactor: amortized KV compaction in a single forward pass.

Wraps one trained :class:`~dexa.compaction.still.perceiver.StillPerceiver` per
layer and exposes them through the :class:`~dexa.compaction.base.Compactor`
interface. Unlike Attention Matching -- which solves a per-context numerical fit
and therefore *needs reference queries* -- STILL has already amortized that work
into the perceiver weights, so :meth:`compact` is a single forward pass and
``needs_ref_queries`` is ``False``.

torch lives only inside this module: the :class:`~dexa.core.types.KVCache` comes
in as numpy float32, is converted to torch for the perceivers, and the resulting
:class:`~dexa.core.types.CompactCache` is converted straight back to numpy.

If no trained perceivers are supplied, identity-initialized perceivers are built
on demand (sized to the requested budget). The forward path is therefore always
runnable -- at the no-compression limit it reproduces the input KV -- which keeps
the whole pipeline testable without any training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dexa.compaction.base import CompactionBudget, Compactor
from dexa.compaction.still.perceiver import StillPerceiver
from dexa.core.types import (
    CompactCache,
    CompactLayer,
    KVCache,
    ModelSpec,
)


class StillCompactor(Compactor):
    """Single-forward-pass compactor backed by per-layer STILL perceivers.

    Parameters
    ----------
    spec:
        Optional :class:`ModelSpec`; if omitted it is read from the cache passed
        to :meth:`compact`.
    perceivers:
        Optional list of trained :class:`StillPerceiver` (one per layer). When
        given, the compaction budget is fixed by their ``n_latents`` and any
        ``budget`` argument is ignored. When ``None``, identity-initialized
        perceivers are created on demand for the requested budget.
    model_rope_theta / internal_rope_theta:
        RoPE bases for lazily-built perceivers (``model_rope_theta`` must match
        the base model that produced the keys).
    device:
        torch device for the perceiver forward pass.
    """

    name = "still"
    needs_ref_queries = False

    def __init__(
        self,
        spec: Optional[ModelSpec] = None,
        perceivers: Optional[list[StillPerceiver]] = None,
        *,
        model_rope_theta: float = 10000.0,
        internal_rope_theta: float = 10000.0,
        device: str = "cpu",
    ) -> None:
        self.spec = spec
        self.perceivers = perceivers
        self.model_rope_theta = float(model_rope_theta)
        self.internal_rope_theta = float(internal_rope_theta)
        self.device = torch.device(device)
        if perceivers is not None:
            for p in perceivers:
                p.to(self.device).eval()
        # cache of lazily-built identity perceivers keyed by (head_dim, t).
        self._lazy: dict[tuple[int, int], list[StillPerceiver]] = {}

    # --- helpers ----------------------------------------------------------
    def _perceivers_for(self, spec: ModelSpec, t: int) -> list[StillPerceiver]:
        if self.perceivers is not None:
            return self.perceivers
        key = (spec.head_dim, t)
        if key not in self._lazy:
            ps = [
                StillPerceiver(
                    head_dim=spec.head_dim,
                    n_latents=t,
                    model_rope_theta=self.model_rope_theta,
                    internal_rope_theta=self.internal_rope_theta,
                ).to(self.device).eval()
                for _ in range(spec.n_layers)
            ]
            self._lazy[key] = ps
        return self._lazy[key]

    # --- public API -------------------------------------------------------
    def compact(
        self,
        cache: KVCache,
        budget: CompactionBudget,
        *,
        ref_queries=None,  # unused: STILL is amortized
    ) -> CompactCache:
        spec = self.spec or cache.spec
        T = cache.seq_len

        if self.perceivers is not None:
            t = min(self.perceivers[0].n_latents, T)
        else:
            t = min(budget.target_t(T), T)
            t = max(1, t)
        perceivers = self._perceivers_for(spec, t)

        positions = torch.from_numpy(cache.positions.astype(np.float32)).to(self.device)

        layers: list[CompactLayer] = []
        with torch.no_grad():
            for li in range(spec.n_layers):
                K = torch.from_numpy(
                    np.ascontiguousarray(cache.layers[li].key, dtype=np.float32)
                ).to(self.device)  # [n_kv, T, d]
                V = torch.from_numpy(
                    np.ascontiguousarray(cache.layers[li].value, dtype=np.float32)
                ).to(self.device)

                Ck, Cv, beta, comp_pos = perceivers[li](K, V, positions)

                Ck_np = Ck.to(torch.float32).cpu().numpy()      # [n_kv, t, d]
                Cv_np = Cv.to(torch.float32).cpu().numpy()
                beta_np = beta.to(torch.float32).cpu().numpy()  # [n_kv, t]
                pos_np = comp_pos.to(torch.float32).cpu().numpy()

                n_kv = Ck_np.shape[0]
                layers.append(
                    CompactLayer(
                        keys=[np.ascontiguousarray(Ck_np[h]) for h in range(n_kv)],
                        values=[np.ascontiguousarray(Cv_np[h]) for h in range(n_kv)],
                        biases=[np.ascontiguousarray(beta_np[h]) for h in range(n_kv)],
                        positions=[pos_np.copy() for _ in range(n_kv)],
                    )
                )

        return CompactCache(
            spec=spec,
            layers=layers,
            logical_length=T,
            method="still",
            meta={
                "target_t": t,
                "trained": self.perceivers is not None,
                "token_ids": list(cache.token_ids) if cache.token_ids else None,
            },
        )
