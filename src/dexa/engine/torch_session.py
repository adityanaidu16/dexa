"""An on-device (GPU-resident) stateful KV session — the path that turns Phase-1's
token savings into real wall-time.

Why this exists. ``HFBackend.recompute_incremental`` is *correct* and reprocesses
far fewer tokens on a mutation, but it keeps the KV in CPU numpy between turns (for
portability + CPU-testability), so every mutation re-uploads the whole prefix cache
to the GPU and converts fp32<->bf16 both ways. On a real 8B that round-trip tax
made incremental *slower* than a full re-prefill despite the token savings.

:class:`TorchKVSession` fixes that: the KV lives as a live ``DynamicCache`` **on the
model's device** across mutations. Mutations touch only the delta:

* ``append(tokens)`` — one forward of the new tokens with the resident cache as
  ``past_key_values`` (the cache grows in place). O(new tokens), no numpy.
* ``edit_suffix(prefix_len, new_tail)`` — truncate the resident cache to
  ``prefix_len`` (an on-device tensor slice) and forward the changed tail. This is
  exact incremental recompute for a mid-context edit, entirely on-device.

Numpy conversion happens only at the boundaries you actually want it:
``to_kvcache()`` (to persist / move a session — the portability wedge) and the
model edge. Nothing is copied to the host on the hot mutation path.

torch lives only in this module (like ``hf_backend``); ``engine/__init__`` does not
import it, so torch stays optional.
"""

from __future__ import annotations

import numpy as np
import torch

from transformers import DynamicCache

from dexa.core.types import KVCache
from dexa.segment.model import SegmentedContext


class TorchKVSession:
    """A GPU-resident KV cache with incremental mutation ops. Built on an
    :class:`~dexa.engine.hf_backend.HFBackend` (reuses its model, device, dtype,
    and attention setup)."""

    def __init__(self, backend, token_ids=None):
        self.be = backend
        self.model = backend.model
        self.device = backend.device
        self.cache = None
        self.token_ids: list[int] = []
        if token_ids:
            self.append(list(token_ids))

    @property
    def n(self) -> int:
        return len(self.token_ids)

    def _ids(self, toks):
        return torch.tensor([list(toks)], dtype=torch.long, device=self.device)

    # --- mutation core (all on-device) ------------------------------------
    def append(self, tokens) -> int:
        """Extend the resident cache by ``tokens`` (incremental prefill of the
        delta). Returns the number of tokens forwarded."""
        tokens = list(tokens)
        if not tokens:
            return 0
        pos = torch.arange(self.n, self.n + len(tokens), device=self.device).unsqueeze(0)
        with torch.no_grad(), self.be._attn_impl("sdpa"):
            out = self.be._kv_forward(input_ids=self._ids(tokens), position_ids=pos,
                                      past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.token_ids += tokens
        return len(tokens)

    def truncate(self, n: int) -> None:
        """Drop everything after position ``n`` — an on-device slice of each layer's
        KV tensors (seq dim). No host copy."""
        if self.cache is None or n >= self.n:
            self.token_ids = self.token_ids[:n]
            return
        pairs = []
        for layer in self.cache.layers:
            pairs.append((layer.keys[:, :, :n, :].contiguous(),
                          layer.values[:, :, :n, :].contiguous()))
        self.cache = DynamicCache(ddp_cache_data=pairs, config=self.model.config)
        self.token_ids = self.token_ids[:n]

    def edit_suffix(self, prefix_len: int, new_tail) -> int:
        """Exact incremental recompute for a mid-context edit: keep the resident
        prefix KV (positions 0..prefix_len), recompute only ``new_tail``. Returns
        the number of tokens forwarded (the recompute cost)."""
        self.truncate(prefix_len)
        return self.append(new_tail)

    def apply(self, prev_ctx: SegmentedContext, new_ctx: SegmentedContext) -> dict:
        """Mutate the session from ``prev_ctx`` to ``new_ctx`` using the segment
        recompute plan — reuse the common prefix, recompute from the first change.
        Assumes the session currently holds ``prev_ctx``."""
        from dexa.segment.plan import plan_incremental
        plan = plan_incremental(prev_ctx, new_ctx, mode="exact")
        prefix = plan.reused_exact_tokens
        recomputed = self.edit_suffix(prefix, new_ctx.token_ids[prefix:])
        return {"reused_tokens": prefix, "recomputed_tokens": recomputed,
                "total_tokens": new_ctx.n_tokens}

    # --- decode / materialize --------------------------------------------
    def greedy(self, max_new_tokens: int) -> list[int]:
        """Greedily decode from the current cache **without** growing the session
        (decodes on a detached copy of the cache)."""
        if self.cache is None:
            raise ValueError("empty session")
        pairs = [(l.keys.clone(), l.values.clone()) for l in self.cache.layers]
        cache = DynamicCache(ddp_cache_data=pairs, config=self.model.config)
        generated: list[int] = []
        cur = self.n
        # prime from the last token's logits: re-run the last token to get logits.
        with torch.no_grad(), self.be._attn_impl("sdpa"):
            last_tok = self.token_ids[-1]
            # truncate the copy by one and feed the last token to get its logit.
            for layer in cache.layers:
                layer.keys = layer.keys[:, :, :-1, :].contiguous()
                layer.values = layer.values[:, :, :-1, :].contiguous()
            pos = torch.tensor([[cur - 1]], device=self.device)
            out = self.model(input_ids=self._ids([last_tok]), position_ids=pos,
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last = out.logits[0, -1]
            for _ in range(max_new_tokens):
                nxt = int(torch.argmax(last).item())
                generated.append(nxt)
                pos = torch.tensor([[cur]], device=self.device)
                out = self.model(input_ids=self._ids([nxt]), position_ids=pos,
                                 past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                last = out.logits[0, -1]
                cur += 1
        return generated

    @classmethod
    def from_kvcache(cls, backend, kv: KVCache) -> "TorchKVSession":
        """Rebuild an on-device session from a portable numpy KVCache (resume /
        rollback boundary)."""
        s = cls(backend)
        s.cache = backend._build_raw_cache(kv)
        s.token_ids = list(kv.token_ids or [])
        return s

    def clone(self) -> "TorchKVSession":
        """A decoupled copy (cloned on-device tensors) — the cheap fork for a
        branch/sub-agent: copy KV, don't re-prefill the shared context."""
        s = TorchKVSession(self.be)
        if self.cache is not None:
            pairs = [(l.keys.clone(), l.values.clone()) for l in self.cache.layers]
            s.cache = DynamicCache(ddp_cache_data=pairs, config=self.model.config)
        s.token_ids = list(self.token_ids)
        return s

    def to_kvcache(self) -> KVCache:
        """Materialize the resident cache to a portable numpy :class:`KVCache` (the
        persistence boundary — only call when you actually save/move the session)."""
        positions = np.arange(self.n, dtype=np.int64)
        return self.be._cache_to_kvcache(self.cache, positions, list(self.token_ids))
