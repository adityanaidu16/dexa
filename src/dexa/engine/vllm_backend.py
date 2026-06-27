"""A vLLM model backend for Dexa, built on vLLM's offline (``LLM``) engine.

This implements :class:`~dexa.engine.base.ModelBackend` against a real vLLM
deployment so Dexa's compaction lifecycle can run on the same high-throughput
engine that serves production traffic. It is the cluster-grade counterpart to
:class:`~dexa.engine.hf_backend.HFBackend`: same interface, same tensor
convention (**numpy float32** at every boundary), but prefill / decode go
through vLLM's paged-attention runtime.

vLLM (and CUDA) live *only* inside this file, and the import is guarded so the
module imports cleanly on machines without vLLM (e.g. a laptop, CI). Constructing
:class:`VLLMBackend` without vLLM installed raises a helpful ``RuntimeError``;
``engine/__init__`` does not import this module, keeping vLLM an optional
dependency (use ``import dexa.engine.vllm_backend``).

How the contract maps onto vLLM
-------------------------------
vLLM does not, out of the box, hand you a layer-by-layer post-RoPE KV tensor or
let you decode against an *externally supplied* compact cache with per-key
additive biases — those are exactly the surfaces Dexa needs. We obtain them with
two mechanisms that exist in current vLLM:

* **KV / query extraction (prefill, reference_queries).** We register forward
  pre-hooks on the model's ``vllm.attention.Attention`` modules. By the time a
  decoder layer calls ``self.attn(q, k, v, ...)`` the rotary embedding has
  already been applied, so the hook observes *post-RoPE* q/k/v — precisely what
  Dexa's convention wants. We run a single prefill step (``max_tokens`` chosen so
  no real decode happens) and collect per-layer K/V (for ``prefill``) or Q (for
  ``reference_queries``), then convert to ``[n_kv_heads, T, head_dim]`` /
  ``[n_q_heads, n_ref, head_dim]`` numpy float32.

  The same data is reachable through vLLM's KV-connector surface
  (``vllm.distributed.kv_transfer`` / ``KVConnectorBase``), which streams the
  paged KV blocks for a finished request; we prefer the hook because it yields
  dense, deduplicated, post-RoPE tensors without paging/block bookkeeping. The
  connector path is sketched in ``_extract_via_kv_connector`` for sites that have
  disaggregated-prefill wired up.

* **Decode against a compact cache (generate, score).** Stock vLLM has no public
  API to (a) seed a sequence's KV cache from arbitrary external tensors and
  (b) add a per-key bias ``beta`` to the pre-softmax scores. Doing this for real
  requires a thin custom attention backend (a ``beta``-aware variant of vLLM's
  attention ``forward``) plus writing the compact K/V into the paged cache via
  the model runner / KV connector. The intended code path is written out and
  clearly marked below; on a cluster you install the small attention shim and it
  runs natively. Until that shim is registered, ``generate`` / ``score`` raise a
  clear ``RuntimeError`` rather than silently returning wrong numbers — the
  faithful eval path remains :class:`HFBackend`, while ``VLLMBackend`` provides
  the production prefill / extraction surface.

Cluster run recipe
------------------
On a GPU node with vLLM installed (``pip install 'vllm>=0.6'``)::

    from dexa.engine.vllm_backend import VLLMBackend
    from dexa.compaction.attention_matching import AttentionMatching  # example
    from dexa.compaction.base import CompactionBudget

    be = VLLMBackend(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,          # set to #GPUs for TP
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        dtype="bfloat16",
    )
    ids = be.tokenize(open("context.txt").read())
    kv  = be.prefill(ids)                # post-RoPE [n_kv_heads, T, head_dim]/layer
    rq  = be.reference_queries(ids, strategy="repeat_prefill", n_per_head=512)
    cc  = AttentionMatching().compact(kv, CompactionBudget(ratio=32), ref_queries=rq)
    # cc is the portable Dexa state object; persist via dexa.core CacheStore.

To enable native compact-decode, register the bias-aware attention backend
(see ``_register_beta_attention``) before constructing the backend::

    VLLMBackend.enable_compact_decode()   # idempotent; no-op if already done
    out = be.generate(cc, prompt_ids, max_new_tokens=64)

Run the structural tests anywhere::

    .venv/bin/python -m pytest tests/test_vllm_backend.py -v
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dexa.core.types import (
    KVCache,
    LayerKV,
    ModelSpec,
    RefQueries,
)
from dexa.engine.base import ContextCache, ModelBackend

# Import-guard vLLM so this module is importable on machines without it (laptop,
# CI). All vLLM/torch usage is funnelled through ``self`` and only happens after
# a successful construction, which itself errors if vLLM is missing.
try:  # pragma: no cover - exercised only on the cluster
    import vllm  # noqa: F401
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
    _VLLM_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # ImportError, or CUDA/driver import failures
    vllm = None  # type: ignore[assignment]
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False
    _VLLM_IMPORT_ERROR = exc


def vllm_available() -> bool:
    """Whether vLLM imported successfully in this process."""
    return _VLLM_AVAILABLE


class VLLMBackend(ModelBackend):
    """:class:`ModelBackend` implemented on vLLM's offline ``LLM`` engine.

    Constructing this requires vLLM (and a GPU); see the module docstring for the
    cluster recipe. The structural contract (ABC surface, numpy boundaries) is
    identical to :class:`HFBackend` and is unit-tested without vLLM present.
    """

    #: set once :meth:`enable_compact_decode` registers the beta-aware backend.
    _COMPACT_DECODE_READY: bool = False

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        seed: int = 0,
        **llm_kwargs: Any,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "VLLMBackend requires the 'vllm' package (and a CUDA GPU), which "
                "is not importable in this environment. Install it on the cluster "
                "with `pip install 'vllm>=0.6'` and run there. Original import "
                f"error: {_VLLM_IMPORT_ERROR!r}. On a laptop/CI use FakeBackend, "
                "or HFBackend for a real-but-CPU model."
            )
        self.model_name = model_name
        self.dtype = dtype

        # Offline engine. ``enforce_eager`` keeps the attention forward in Python
        # so our extraction hooks (and the optional beta shim) actually fire;
        # CUDA-graph capture would bypass module-level hooks.
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            seed=seed,
            enforce_eager=True,
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()

        cfg = self._hf_config()
        n_q = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
        self._spec = ModelSpec(
            name=model_name,
            n_layers=cfg.num_hidden_layers,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_size=cfg.hidden_size,
            dtype=dtype,
        )

    # --- engine introspection helpers -------------------------------------
    def _hf_config(self):
        """The underlying HF ``PretrainedConfig`` vLLM loaded."""
        return self.llm.llm_engine.model_config.hf_config

    def _model_runner(self):
        """The worker-local model runner (single-GPU / driver worker)."""
        engine = self.llm.llm_engine
        # vLLM has moved this attribute around across versions; probe the known
        # locations. On TP>1 only the driver worker is reachable in-process; KV
        # extraction then goes through the KV-connector path instead.
        executor = getattr(engine, "model_executor", None) or getattr(engine, "engine", None)
        for path in ("driver_worker", "_driver_worker"):
            worker = getattr(executor, path, None)
            if worker is not None:
                return getattr(worker, "model_runner", None) or getattr(worker, "_model_runner")
        raise RuntimeError("could not locate vLLM model runner for KV extraction")

    def _attention_modules(self) -> list:
        """The ordered list of ``vllm.attention.Attention`` modules (one per
        decoder layer), used as hook anchors for post-RoPE q/k/v capture."""
        from vllm.attention import Attention  # local import: vllm-only symbol

        model = self._model_runner().model
        mods = [m for m in model.modules() if isinstance(m, Attention)]
        if len(mods) != self._spec.n_layers:
            # Some architectures interleave non-attention layers; fall back to a
            # name sort so the order matches the decoder stack.
            named = [
                (name, m)
                for name, m in model.named_modules()
                if isinstance(m, Attention)
            ]
            named.sort(key=lambda kv: kv[0])
            mods = [m for _, m in named]
        return mods

    # --- properties -------------------------------------------------------
    @property
    def spec(self) -> ModelSpec:
        return self._spec

    # --- tokenization -----------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # --- internal: post-RoPE q/k/v capture --------------------------------
    def _capture_qkv(self, token_ids: list[int], *, position_offset: int = 0) -> dict:
        """Run a single prefill of ``token_ids`` and capture per-layer post-RoPE
        q/k/v as numpy arrays.

        Returns a dict with keys ``q`` / ``k`` / ``v``: lists (per layer) of
        ``[n_heads, T, head_dim]`` float32 (q has ``n_q_heads``; k/v have
        ``n_kv_heads``).

        Mechanism: a forward *pre-hook* on each ``Attention`` module records its
        (q, k, v) positional inputs, which are already rotary-embedded by the
        decoder layer. We then ask vLLM for one token of greedy output, which
        forces exactly one prefill pass over the prompt. This is the real,
        cluster-runnable path; it cannot execute here because vLLM is absent.
        """
        import torch  # vllm pulls in torch; safe to import on the cluster

        s = self._spec
        mods = self._attention_modules()
        captured: list[dict] = [dict() for _ in range(len(mods))]
        handles = []

        def make_hook(li: int):
            def hook(_module, args, _kwargs=None):
                # vLLM Attention.forward signature is (query, key, value, ...);
                # tensors are flattened [num_tokens, n_heads*head_dim].
                q, k, v = args[0], args[1], args[2]
                captured[li]["q"] = q.detach()
                captured[li]["k"] = k.detach()
                captured[li]["v"] = v.detach()
            return hook

        for li, m in enumerate(mods):
            handles.append(m.register_forward_pre_hook(make_hook(li), with_kwargs=True))

        try:
            # One forward over the prompt; max_tokens=1 => prefill + a single
            # (discarded) decode step. ``prompt_token_ids`` skips re-tokenizing.
            params = SamplingParams(max_tokens=1, temperature=0.0)
            self.llm.generate(
                {"prompt_token_ids": list(token_ids)},
                params,
                use_tqdm=False,
            )
        finally:
            for h in handles:
                h.remove()

        T = len(token_ids)

        def reshape(t, n_heads: int) -> np.ndarray:
            # [num_tokens, n_heads*head_dim] -> [n_heads, T, head_dim].
            arr = t.to(torch.float32).cpu().numpy()
            arr = arr.reshape(arr.shape[0], n_heads, s.head_dim)[:T]
            return np.ascontiguousarray(arr.transpose(1, 0, 2))

        return {
            "q": [reshape(captured[li]["q"], s.n_q_heads) for li in range(s.n_layers)],
            "k": [reshape(captured[li]["k"], s.n_kv_heads) for li in range(s.n_layers)],
            "v": [reshape(captured[li]["v"], s.n_kv_heads) for li in range(s.n_layers)],
        }

    def _extract_via_kv_connector(self, token_ids: list[int]) -> dict:  # pragma: no cover
        """Alternative KV extraction through vLLM's KV-connector surface.

        Intended for TP>1 / disaggregated-prefill deployments where the dense
        per-layer tensors are not reachable in-process. The connector
        (``vllm.distributed.kv_transfer.kv_connector``) streams the finished
        request's paged KV blocks; we gather the blocks for the sequence, undo
        the paging (block table -> contiguous token order), and stack per layer.
        Wiring depends on the connector configured at engine init, so this is a
        documented hook rather than a one-size implementation.
        """
        raise RuntimeError(
            "KV-connector extraction requires a connector configured at engine "
            "init (kv_transfer_config). Use the default hook-based path "
            "(_capture_qkv) for single-process TP=1 deployments."
        )

    # --- prefill ----------------------------------------------------------
    def prefill(self, token_ids: list[int], *, position_offset: int = 0) -> KVCache:
        """Run vLLM prefill and return the full post-RoPE KV cache as numpy.

        ``position_offset`` is recorded in the returned ``positions`` for RoPE /
        logical-length bookkeeping downstream; vLLM always prefixes from position
        0 internally, so a non-zero offset only relabels the stored positions
        (the keys are already post-RoPE at the engine's own positions).
        """
        token_ids = list(token_ids)
        T = len(token_ids)
        s = self._spec
        qkv = self._capture_qkv(token_ids, position_offset=position_offset)
        layers: list[LayerKV] = []
        for li in range(s.n_layers):
            layers.append(LayerKV(key=qkv["k"][li], value=qkv["v"][li]))
        positions = np.arange(position_offset, position_offset + T, dtype=np.int64)
        return KVCache(spec=s, layers=layers, positions=positions, token_ids=token_ids)

    # --- reference queries ------------------------------------------------
    #: fixed self-study prompts (mirrors HFBackend): synthetic "questions" whose
    #: post-RoPE queries cover how a real follow-up attends back over context.
    _SELF_STUDY_PROMPTS = (
        "Summarize the key facts.",
        "List every specific detail mentioned.",
        "What questions could be asked about this?",
        "Repeat the important numbers and names.",
    )

    def reference_queries(
        self,
        token_ids: list[int],
        *,
        strategy: str = "repeat_prefill",
        n_per_head: int = 512,
    ) -> RefQueries:
        """Produce post-RoPE reference queries for attention-matching compaction.

        Strategies match :class:`HFBackend`:
          * ``self`` — capture the context's own post-RoPE queries.
          * ``repeat_prefill`` — capture queries at a *second* occurrence of the
            context (after a separator), whose RoPE phase resembles a follow-up.
          * ``self_study`` — generate short continuations under fixed prompts and
            capture queries at the prompt+generated positions.
        Queries are subsampled to ``n_per_head`` along the query axis.
        """
        token_ids = list(token_ids)
        T = len(token_ids)
        s = self._spec

        if strategy == "self":
            qkv = self._capture_qkv(token_ids)
            layers = [self._subsample(qkv["q"][li], n_per_head) for li in range(s.n_layers)]
            return RefQueries(spec=s, layers=layers)

        if strategy == "repeat_prefill":
            sep = [self._sep_token()]
            full = token_ids + sep + token_ids
            ref_start = T + len(sep)
            ref_slice = slice(ref_start, ref_start + T)
        elif strategy == "self_study":
            return self._self_study_queries(token_ids, n_per_head=n_per_head)
        else:
            raise ValueError(f"unknown reference-query strategy: {strategy!r}")

        qkv = self._capture_qkv(full)
        layers = [
            self._subsample(qkv["q"][li][:, ref_slice, :], n_per_head)
            for li in range(s.n_layers)
        ]
        return RefQueries(spec=s, layers=layers)

    def _self_study_queries(self, token_ids: list[int], *, n_per_head: int) -> RefQueries:
        """Generate short continuations under fixed prompts (greedy, via vLLM)
        and capture post-RoPE queries at the prompt+generated positions. These
        sit just after the context (logical positions ``T..``), matching a real
        follow-up question's RoPE phase. Aggregate across prompts, then subsample.
        """
        s = self._spec
        T = len(token_ids)
        sep = [self._sep_token()]
        n_gen = 16

        chunks: list[list[np.ndarray]] = [[] for _ in range(s.n_layers)]
        for prompt in self._SELF_STUDY_PROMPTS:
            p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            primer = token_ids + sep + list(p_ids)
            gen = self._greedy_continuation(primer, n_gen)
            full = primer + gen
            ref_slice = slice(T + len(sep), len(full))
            qkv = self._capture_qkv(full)
            for li in range(s.n_layers):
                chunks[li].append(qkv["q"][li][:, ref_slice, :])

        layers: list[np.ndarray] = []
        for li in range(s.n_layers):
            q = np.concatenate(chunks[li], axis=1)
            layers.append(self._subsample(q, n_per_head))
        return RefQueries(spec=s, layers=layers)

    def _greedy_continuation(self, prefix: list[int], n_new: int) -> list[int]:
        """Greedily decode ``n_new`` tokens after ``prefix`` using vLLM."""
        params = SamplingParams(max_tokens=n_new, temperature=0.0)
        out = self.llm.generate(
            {"prompt_token_ids": list(prefix)}, params, use_tqdm=False
        )
        return list(out[0].outputs[0].token_ids)

    @staticmethod
    def _subsample(q: np.ndarray, n_per_head: int) -> np.ndarray:
        """Evenly subsample the query axis (axis=1) to at most ``n_per_head``."""
        if q.shape[1] > n_per_head:
            idx = np.linspace(0, q.shape[1] - 1, n_per_head).astype(int)
            q = q[:, idx]
        return np.ascontiguousarray(q)

    def _sep_token(self) -> int:
        for attr in ("bos_token_id", "eos_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                return int(tid)
        return 0

    # --- optional native compact-decode shim ------------------------------
    @classmethod
    def enable_compact_decode(cls) -> None:  # pragma: no cover - cluster only
        """Register the ``beta``-aware attention backend so :meth:`generate` /
        :meth:`score` can decode against an externally-supplied compact cache.

        This installs :func:`_register_beta_attention` once. It is a no-op if
        already installed or if vLLM is absent (construction would have failed
        first). Idempotent.
        """
        if cls._COMPACT_DECODE_READY:
            return
        if not _VLLM_AVAILABLE:
            raise RuntimeError("enable_compact_decode requires vllm")
        _register_beta_attention()
        cls._COMPACT_DECODE_READY = True

    def _require_compact_decode(self) -> None:
        if not type(self)._COMPACT_DECODE_READY:
            raise RuntimeError(
                "VLLMBackend.generate/score against a (compact) cache needs the "
                "bias-aware attention backend. Call "
                "VLLMBackend.enable_compact_decode() once before decoding, or use "
                "HFBackend for the faithful CPU eval path. Stock vLLM has no "
                "public API to seed external KV + per-key biases."
            )

    # --- decode / scoring against a (possibly compact) cache --------------
    def generate(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 64,
        greedy: bool = True,
    ) -> list[int]:
        """Decode a continuation attending over ``context`` then ``prompt``.

        Real intended path (requires :meth:`enable_compact_decode`):
          1. Seed a fresh vLLM sequence's paged KV cache from ``context`` (full
             or compact) via the model runner / KV connector, padding ragged
             per-head compact budgets to a common physical length.
          2. Attach the per-layer, per-key additive bias ``beta`` to the
             sequence so the registered attention backend adds it pre-softmax
             (full caches use ``beta=0``; compact caches use the stored biases;
             padded slots use ``-inf``).
          3. Run ``llm.generate`` with ``position_ids`` starting at the cache's
             logical length so RoPE phases stay correct, and return the new ids.

        Because steps 1-2 use the optional beta-aware backend, this raises a
        clear error until that shim is registered (see module docstring).
        """
        self._require_compact_decode()
        return self._decode_with_injected_cache(  # pragma: no cover - cluster only
            context, list(prompt_token_ids), max_new_tokens=max_new_tokens, greedy=greedy
        )

    def score(
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
    ) -> np.ndarray:
        """Teacher-forced per-token log-probs of ``target`` given
        ``context`` + ``prompt``; shape ``[len(target)]``.

        Same injected-cache mechanism as :meth:`generate` (see there). vLLM
        exposes per-token logprobs via ``SamplingParams(prompt_logprobs=...)``;
        the intended path requests logprobs for the ``target`` positions of a
        ``prompt+target`` forward over the injected ``context`` and gathers the
        chosen-token logprobs. Requires :meth:`enable_compact_decode`.
        """
        target_token_ids = list(target_token_ids)
        if not target_token_ids:
            return np.zeros(0, dtype=np.float32)
        self._require_compact_decode()
        return self._score_with_injected_cache(  # pragma: no cover - cluster only
            context, list(prompt_token_ids), target_token_ids
        )

    # The two methods below are the real cluster code paths for decode/score
    # against an injected (compact) cache. They are only reachable once
    # ``enable_compact_decode`` has registered the beta-aware backend, so they
    # never run in the vllm-less environment; they are written out so a cluster
    # deployment has the concrete integration to finish wiring.
    def _decode_with_injected_cache(  # pragma: no cover - cluster only
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        greedy: bool,
    ) -> list[int]:
        seq = _InjectedSequence.from_context(self, context)
        seq.attach_to_runner(self._model_runner())
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0 if greedy else 1.0,
        )
        prompt = prompt_token_ids or [seq.boundary_token]
        out = self.llm.generate(
            {"prompt_token_ids": prompt},
            params,
            use_tqdm=False,
            # custom field consumed by the beta-aware backend to bind this
            # request to the pre-seeded KV + biases and logical position offset.
            extra_request_args={"dexa_injected_kv": seq.handle},
        )
        return list(out[0].outputs[0].token_ids)

    def _score_with_injected_cache(  # pragma: no cover - cluster only
        self,
        context: ContextCache,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
    ) -> np.ndarray:
        seq = _InjectedSequence.from_context(self, context)
        seq.attach_to_runner(self._model_runner())
        n_prompt = len(prompt_token_ids)
        if n_prompt:
            full = prompt_token_ids + target_token_ids
            first = n_prompt  # first target position in ``full``
        else:
            full = [seq.boundary_token] + target_token_ids
            first = 1
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=0,  # logprob of each prompt token under the model
        )
        out = self.llm.generate(
            {"prompt_token_ids": full},
            params,
            use_tqdm=False,
            extra_request_args={"dexa_injected_kv": seq.handle},
        )
        plps = out[0].prompt_logprobs  # list aligned to ``full`` (index 0 is None)
        scores = np.empty(len(target_token_ids), dtype=np.float32)
        for k, tok in enumerate(target_token_ids):
            scores[k] = plps[first + k][tok].logprob
        return scores

    # --- optional attention outputs (numpy, model-free) -------------------
    def attention_outputs(self, cache: ContextCache, queries: RefQueries) -> list[np.ndarray]:
        """Per-layer locally-normalized attention output of ``queries`` over
        ``cache``. Pure numpy (model-free), identical to the other backends, so
        compactor quality checks run without touching the GPU."""
        s = self._spec
        scale = 1.0 / np.sqrt(s.head_dim)
        out: list[np.ndarray] = []
        for li in range(s.n_layers):
            Q = queries.layers[li]
            n_ref = Q.shape[1]
            res = np.zeros((s.n_q_heads, n_ref, s.head_dim), dtype=np.float32)
            for qh in range(s.n_q_heads):
                h = s.kv_head_of(qh)
                if isinstance(cache, KVCache):
                    K = cache.layers[li].key[h]
                    V = cache.layers[li].value[h]
                    beta = None
                else:
                    K = cache.layers[li].keys[h]
                    V = cache.layers[li].values[h]
                    beta = cache.layers[li].biases[h]
                logits = (Q[qh] @ K.T) * scale
                if beta is not None:
                    logits = logits + beta[None, :]
                logits -= logits.max(axis=-1, keepdims=True)
                w = np.exp(logits)
                w /= np.clip(w.sum(axis=-1, keepdims=True), 1e-8, None)
                res[qh] = w @ V
            out.append(res)
        return out


# --- optional beta-aware attention backend (registered on the cluster) ----
def _register_beta_attention() -> None:  # pragma: no cover - cluster only
    """Install a ``beta``-aware attention backend into vLLM.

    The shim subclasses vLLM's current attention backend and, in ``forward``,
    adds a per-request, per-layer, per-key additive bias tensor (Dexa's
    ``beta``) to the pre-softmax logits before the softmax — the exact analogue
    of :func:`dexa.engine.hf_backend._dexa_eager_attention_forward`. It also
    teaches the model runner to seed a sequence's paged KV blocks from external
    tensors (the compact K/V) and to start position ids at the cache's logical
    length. Registration is via vLLM's attention-backend selector
    (``VLLM_ATTENTION_BACKEND`` / ``vllm.attention.selector``); the concrete
    subclass is deployment-specific (FlashAttention vs. XFormers vs. Triton),
    which is why it is wired here at install time rather than imported eagerly.
    """
    raise RuntimeError(
        "The beta-aware vLLM attention backend is deployment-specific (depends "
        "on the attention impl your cluster runs) and must be provided as a "
        "small site shim. See the docstring of _register_beta_attention for the "
        "exact integration points (selector, paged-KV seeding, position offset)."
    )


class _InjectedSequence:  # pragma: no cover - cluster only
    """Handle binding a Dexa ``context`` (full or compact KV + biases) to a vLLM
    request so the beta-aware backend can decode against it.

    Construction packs the per-layer K/V (padded to a common physical length for
    ragged compact budgets) and per-key ``beta`` into the layout the registered
    backend expects, and records the logical length (for the position offset) and
    a boundary token (used when the caller's prompt is empty, mirroring
    :meth:`HFBackend._drop_last`). ``attach_to_runner`` writes the KV into free
    paged blocks and returns an opaque ``handle`` referenced by the request.
    """

    def __init__(self, handle: str, boundary_token: int) -> None:
        self.handle = handle
        self.boundary_token = boundary_token

    @classmethod
    def from_context(cls, backend: "VLLMBackend", context: ContextCache) -> "_InjectedSequence":
        raise RuntimeError(
            "_InjectedSequence.from_context is realized by the site-specific "
            "beta-aware backend shim (see _register_beta_attention)."
        )

    def attach_to_runner(self, model_runner) -> None:
        raise RuntimeError(
            "_InjectedSequence.attach_to_runner is realized by the site-specific "
            "beta-aware backend shim (see _register_beta_attention)."
        )
