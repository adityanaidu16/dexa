"""Model backend abstraction.

A backend is whatever can (a) turn text into a KV cache, (b) produce reference
queries for attention-matching, and (c) decode/score against either a full or a
compacted cache. The HF backend implements this against a real transformer;
:class:`~dexa.engine.fake.FakeBackend` implements it with deterministic numpy
attention for tests and CI (no torch, no GPU).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from dexa.core.types import CompactCache, KVCache, ModelSpec, RefQueries

# A context the model decodes against may be raw or compacted.
ContextCache = Union[KVCache, CompactCache]


class ModelBackend(ABC):
    """Interface every engine integration implements."""

    @property
    @abstractmethod
    def spec(self) -> ModelSpec: ...

    # --- tokenization -----------------------------------------------------
    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @abstractmethod
    def detokenize(self, token_ids: list[int]) -> str: ...

    # --- prefill / KV extraction -----------------------------------------
    @abstractmethod
    def prefill(self, token_ids: list[int], *, position_offset: int = 0) -> KVCache:
        """Run prefill and return the full KV cache (keys post-RoPE)."""

    @abstractmethod
    def reference_queries(
        self,
        token_ids: list[int],
        *,
        strategy: str = "repeat_prefill",
        n_per_head: int = 512,
    ) -> RefQueries:
        """Produce reference queries for attention-matching compaction.

        ``strategy`` is one of ``repeat_prefill`` (cheap, no generation),
        ``self_study`` (generate continuations), or ``self`` (use the context's
        own queries). Implementations may subsample to ``n_per_head``.
        """

    # --- decode / scoring against a (possibly compact) cache --------------
    @abstractmethod
    def generate(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 64,
        greedy: bool = True,
    ) -> list[int]:
        """Decode a continuation, attending over ``context`` then ``prompt``.

        ``context`` may be a full :class:`KVCache` or a :class:`CompactCache`;
        for compact caches the per-key biases must be injected additively into
        the attention scores. New tokens get positions starting at the cache's
        logical length.
        """

    @abstractmethod
    def score(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
    ) -> np.ndarray:
        """Teacher-forced per-token log-probabilities of ``target`` given
        ``context`` + ``prompt``. Returns shape [len(target)]."""

    # --- optional: stateful session primitives (override to enable serving) --
    def extend(self, context, token_ids: list[int]):
        """Append tokens to a session's KV cache (incremental prefill of the
        delta). Returns the grown KVCache. Override to support stateful serving."""
        raise NotImplementedError

    def generate_and_extend(self, context, prompt_token_ids: list[int], *,
                            max_new_tokens: int = 64, greedy: bool = True):
        """Decode a response against a session cache and return (response_tokens,
        grown KVCache including prompt + response). Override to support serving."""
        raise NotImplementedError

    # --- optional: direct attention output (used by compaction unit tests)-
    def attention_outputs(self, cache: ContextCache, queries: RefQueries) -> list[np.ndarray]:
        """Per-layer locally-normalized attention output of ``queries`` over
        ``cache``: softmax(Q K^T [+ beta]) V. Shape per layer
        [n_q_heads, n_ref, head_dim]. Optional; backends that can compute it
        enable cheap, model-free quality checks of a compactor."""
        raise NotImplementedError
