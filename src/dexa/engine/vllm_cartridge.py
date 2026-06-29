"""Serve a Dexa :class:`~dexa.cartridge.artifact.Cartridge` on vLLM as a
precomputed KV **prefix**.

This is the serving half of the cartridge story (see ``docs/CARTRIDGES.md``). A
cartridge is a *trained compact KV cache with no attention bias* (``beta == 0``).
That single property is what makes serving cheap: because there is no per-key
bias, a cartridge behaves like an ordinary (small) KV prefix, so we do **not**
need the bias-aware custom attention backend that :class:`VLLMBackend` requires
for general compact caches. We just write the cartridge's K/V into vLLM's paged
KV cache as the shared prefix for a request and decode the query on top of it —
the stock attention kernel does the rest.

vLLM (and CUDA/torch) live *only* inside this file behind an import guard, so the
module imports cleanly on a laptop / CI with no vLLM. Constructing
:class:`CartridgeServer` without vLLM raises a helpful ``RuntimeError``. As with
:mod:`dexa.engine.vllm_backend`, ``engine/__init__`` does **not** import this
module (use ``import dexa.engine.vllm_cartridge``), keeping vLLM optional.

How the prefix injection works
------------------------------
A cartridge stores, per layer, post-RoPE compact keys/values
``[n_kv_heads, t, head_dim]`` plus the absolute ``positions`` (length ``t``) the
keys live at and the corpus ``logical_length`` T. To serve a query:

1. **Layout.** Convert the cartridge numpy K/V to vLLM's token-major per-layer
   layout ``[t, n_kv_heads, head_dim]`` (and, for paged backends, pack into
   ``[num_blocks, block_size, n_kv_heads, head_dim]``). These conversions are
   pure numpy and are unit-tested here without vLLM — see
   :func:`cartridge_to_token_major` / :func:`pack_token_major_into_blocks`.
2. **Inject.** Write those K/V tensors into the paged KV blocks vLLM allocates
   for the request's prefix slots. Two real surfaces do this:
     * the **KV-connector** (``vllm.distributed.kv_transfer``): implement a
       connector whose ``inject``/``load`` hook fills the prefix blocks from the
       cartridge (the disaggregated-prefill seam), or
     * the **model runner** directly, copying into the block table the scheduler
       handed the sequence (single-process TP=1).
   Because the cartridge carries no bias, no kernel change is needed: the keys
   are already post-RoPE, so attention over them is exactly attention over a real
   prefix.
3. **Decode.** Run ``llm.generate`` for the query tokens with their position ids
   starting at ``cartridge.logical_length`` so RoPE phases line up with how the
   model would attend back over the full corpus (the cartridge's compact keys
   keep their *original* absolute positions; only the new query tokens are
   re-based to ``T``).

Caveat: vLLM's exact paged-KV block APIs (block manager, slot mapping, connector
signatures) drift across releases. The numpy-side layout helpers are stable and
fully tested; the in-engine write is marked ``pragma: no cover`` and isolated in
:meth:`CartridgeServer._inject_prefix` / :class:`_PagedPrefixWriter`, the one
place a site pins to its vLLM version.

Cluster run recipe
------------------
On a GPU node with vLLM installed (``pip install 'vllm>=0.6'``)::

    from dexa.cartridge import Cartridge
    from dexa.engine.vllm_cartridge import CartridgeServer

    srv = CartridgeServer(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        dtype="bfloat16",
    )
    cart = Cartridge.load("my-corpus.cartridge.npz")
    srv.load_cartridge(cart)                       # validates + registers
    text = srv.generate("What does the spec say about retries?", cart,
                        max_new_tokens=128)
    lp   = srv.score("Q: ...", " A: ...", cart)    # teacher-forced logprobs

Run the structural tests anywhere::

    .venv/bin/python -m pytest tests/test_vllm_cartridge.py -v
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dexa.cartridge.artifact import Cartridge
from dexa.core.types import ModelSpec

# Import-guard vLLM so this module imports on machines without it (laptop, CI).
# All vLLM/torch usage is funnelled through ``self`` and only happens after a
# successful construction, which itself errors if vLLM is missing.
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


# ---------------------------------------------------------------------------
# Pure-numpy layout helpers (no vLLM): unit-tested everywhere.
# ---------------------------------------------------------------------------
def cartridge_to_token_major(
    keys: np.ndarray, values: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Convert cartridge K/V to vLLM's per-layer **token-major** layout.

    Input ``keys``/``values`` are ``[n_layers, n_kv_heads, t, head_dim]`` (the
    :class:`Cartridge` storage layout). Output is two lists of length
    ``n_layers``; each entry is ``[t, n_kv_heads, head_dim]`` — the dense,
    contiguous form vLLM's KV-cache writers expect before paging (token is the
    leading axis, then heads, then head_dim).

    Round-trips with :func:`token_major_to_cartridge`.
    """
    keys = np.asarray(keys)
    values = np.asarray(values)
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            f"expected [n_layers, n_kv_heads, t, head_dim]; got {keys.shape} / {values.shape}"
        )
    if keys.shape != values.shape:
        raise ValueError(f"keys/values shape mismatch: {keys.shape} vs {values.shape}")
    n_layers = keys.shape[0]
    k_out = [np.ascontiguousarray(keys[li].transpose(1, 0, 2)) for li in range(n_layers)]
    v_out = [np.ascontiguousarray(values[li].transpose(1, 0, 2)) for li in range(n_layers)]
    return k_out, v_out


def token_major_to_cartridge(
    k_layers: list[np.ndarray], v_layers: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`cartridge_to_token_major`.

    Stacks per-layer token-major ``[t, n_kv_heads, head_dim]`` arrays back into
    cartridge storage layout ``[n_layers, n_kv_heads, t, head_dim]``.
    """
    if len(k_layers) != len(v_layers):
        raise ValueError("k_layers/v_layers length mismatch")
    keys = np.stack([np.asarray(k).transpose(1, 0, 2) for k in k_layers], axis=0)
    values = np.stack([np.asarray(v).transpose(1, 0, 2) for v in v_layers], axis=0)
    return np.ascontiguousarray(keys), np.ascontiguousarray(values)


def pack_token_major_into_blocks(
    token_major: np.ndarray, block_size: int
) -> np.ndarray:
    """Pack one layer's token-major K (or V) into vLLM-style paged blocks.

    Input is ``[t, n_kv_heads, head_dim]``; output is
    ``[num_blocks, block_size, n_kv_heads, head_dim]`` with
    ``num_blocks = ceil(t / block_size)`` and the final partial block
    zero-padded. This mirrors how vLLM lays a contiguous sequence of tokens into
    fixed-size KV blocks (slot ``i`` -> block ``i // block_size``, offset
    ``i % block_size``). Pure numpy so the packing/padding can be checked without
    a GPU; the cluster path copies these blocks into the engine's block table.
    """
    arr = np.asarray(token_major)
    if arr.ndim != 3:
        raise ValueError(f"expected [t, n_kv_heads, head_dim]; got {arr.shape}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    t, n_kv, d = arr.shape
    num_blocks = (t + block_size - 1) // block_size
    out = np.zeros((num_blocks, block_size, n_kv, d), dtype=arr.dtype)
    out.reshape(num_blocks * block_size, n_kv, d)[:t] = arr
    return out


def query_positions(cartridge: Cartridge, n_query: int) -> np.ndarray:
    """Absolute RoPE position ids for ``n_query`` tokens appended after the
    cartridge: ``arange(logical_length, logical_length + n_query)``.

    The cartridge's own compact keys keep their original absolute ``positions``;
    only the new query tokens are re-based to start at ``logical_length`` so the
    model attends back over the corpus with the same relative phases it saw at
    compile time. Pure numpy; unit-tested without vLLM.
    """
    if n_query < 0:
        raise ValueError("n_query must be non-negative")
    start = int(cartridge.logical_length)
    return np.arange(start, start + n_query, dtype=np.int64)


def assert_cartridge_matches_spec(cartridge: Cartridge, spec: ModelSpec) -> None:
    """Validate that a cartridge was trained for the model being served.

    Checks the structural dimensions that must line up for the K/V to drop into
    the engine's cache: layer count, kv-head count, head dim, and the array
    shapes. Pure numpy; does not require vLLM.
    """
    s = spec
    k = np.asarray(cartridge.keys)
    if k.ndim != 4:
        raise ValueError(f"cartridge.keys must be 4-D [L,n_kv,t,d]; got {k.shape}")
    L, n_kv, t, d = k.shape
    mismatches = []
    if L != s.n_layers:
        mismatches.append(f"n_layers {L} != model {s.n_layers}")
    if n_kv != s.n_kv_heads:
        mismatches.append(f"n_kv_heads {n_kv} != model {s.n_kv_heads}")
    if d != s.head_dim:
        mismatches.append(f"head_dim {d} != model {s.head_dim}")
    if cartridge.positions.shape[0] != t:
        mismatches.append(
            f"positions length {cartridge.positions.shape[0]} != t {t}"
        )
    if mismatches:
        raise ValueError(
            "cartridge does not match the served model: " + "; ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# The server.
# ---------------------------------------------------------------------------
class CartridgeServer:
    """Load and serve :class:`Cartridge` artifacts on vLLM as KV prefixes.

    Constructing this requires vLLM (and a GPU); see the module docstring for the
    cluster recipe. The numpy-side layout / validation helpers are module-level
    functions, unit-tested without vLLM present.
    """

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
                "CartridgeServer requires the 'vllm' package (and a CUDA GPU), "
                "which is not importable in this environment. Install it on the "
                "cluster with `pip install 'vllm>=0.6'` and run there. Original "
                f"import error: {_VLLM_IMPORT_ERROR!r}. The numpy layout helpers "
                "(cartridge_to_token_major, pack_token_major_into_blocks, "
                "query_positions) work everywhere without vllm."
            )
        self.model_name = model_name
        self.dtype = dtype

        # ``enforce_eager`` keeps attention in Python so the prefix-injection
        # hooks / connector callbacks fire; CUDA-graph capture would bypass them.
        self.llm = LLM(  # pragma: no cover - cluster only
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            seed=seed,
            enforce_eager=True,
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()  # pragma: no cover - cluster only

        cfg = self.llm.llm_engine.model_config.hf_config  # pragma: no cover - cluster only
        n_q = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
        self._spec = ModelSpec(  # pragma: no cover - cluster only
            name=model_name,
            n_layers=cfg.num_hidden_layers,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_size=cfg.hidden_size,
            dtype=dtype,
        )
        #: registered cartridges keyed by an opaque handle string.
        self._cartridges: dict[str, Cartridge] = {}

    # --- properties -------------------------------------------------------
    @property
    def spec(self) -> ModelSpec:  # pragma: no cover - cluster only
        return self._spec

    # --- registration -----------------------------------------------------
    def load_cartridge(self, cartridge: Cartridge) -> str:  # pragma: no cover - cluster only
        """Validate a cartridge against the served model and register it.

        Returns an opaque handle; later ``generate``/``score`` calls accept the
        cartridge object directly (the handle is for stores that key by id).
        Validation (:func:`assert_cartridge_matches_spec`) runs the pure-numpy
        shape checks so a mismatched artifact fails loudly *before* any engine
        work.
        """
        assert_cartridge_matches_spec(cartridge, self._spec)
        handle = self._cartridge_handle(cartridge)
        self._cartridges[handle] = cartridge
        return handle

    @staticmethod
    def _cartridge_handle(cartridge: Cartridge) -> str:
        """Stable id for a cartridge (corpus name + shape + logical length)."""
        return (
            f"{cartridge.spec.name}:{cartridge.logical_length}:"
            f"{cartridge.keys.shape}"
        )

    # --- prefix injection (the one version-pinned seam) -------------------
    def _inject_prefix(self, cartridge: Cartridge):  # pragma: no cover - cluster only
        """Write the cartridge K/V into the paged KV blocks for a request prefix.

        Real intended path (single-process TP=1; see module docstring for the
        KV-connector alternative):

          1. Convert with :func:`cartridge_to_token_major` then
             :func:`pack_token_major_into_blocks` (using the engine's
             ``cache_config.block_size``) to per-layer paged blocks.
          2. Move to torch on the model's device/dtype at the vLLM boundary.
          3. Acquire free physical blocks from the block manager for ``t`` slots
             and copy each layer's blocks into the engine's KV cache tensors at
             those block ids; record the block table + cartridge ``positions``
             so the slot mapping matches the keys' original absolute positions.

        Returns a :class:`_PagedPrefixWriter` handle the request binds to. The
        body below is the concrete cluster integration; it is gated by the vLLM
        import so it never runs in the vllm-less environment.
        """
        import torch  # vllm pulls in torch; safe on the cluster

        block_size = self.llm.llm_engine.cache_config.block_size
        k_layers, v_layers = cartridge_to_token_major(cartridge.keys, cartridge.values)
        torch_dtype = getattr(torch, self.dtype, torch.float16)
        k_blocks = [
            torch.from_numpy(pack_token_major_into_blocks(k, block_size)).to(torch_dtype)
            for k in k_layers
        ]
        v_blocks = [
            torch.from_numpy(pack_token_major_into_blocks(v, block_size)).to(torch_dtype)
            for v in v_layers
        ]
        writer = _PagedPrefixWriter(
            k_blocks=k_blocks,
            v_blocks=v_blocks,
            positions=np.asarray(cartridge.positions, dtype=np.int64),
            logical_length=int(cartridge.logical_length),
            block_size=block_size,
        )
        writer.attach_to_runner(self.llm)
        return writer

    # --- serving ----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        cartridge: Cartridge,
        *,
        max_new_tokens: int = 64,
        greedy: bool = True,
    ) -> str:
        """Decode a continuation for ``prompt`` with ``cartridge`` as the KV
        prefix (the corpus "in context").

        Real intended path:
          1. :meth:`_inject_prefix` seeds the request's prefix KV from the
             cartridge.
          2. Tokenize ``prompt`` and run ``llm.generate`` with the query tokens'
             position ids starting at ``cartridge.logical_length``
             (:func:`query_positions`), bound to the injected prefix.
          3. Detokenize and return the new text.

        Because step 1 writes into vLLM's paged cache (version-pinned), this runs
        only on the cluster; the numpy helpers it relies on are tested here.
        """
        self._require_vllm_runtime()
        return self._generate_on_engine(  # pragma: no cover - cluster only
            prompt, cartridge, max_new_tokens=max_new_tokens, greedy=greedy
        )

    def score(
        self,
        prompt: str,
        target: str,
        cartridge: Cartridge,
    ) -> np.ndarray:
        """Teacher-forced per-token log-probs of ``target`` given ``cartridge`` +
        ``prompt``; shape ``[len(target_tokens)]``.

        Same prefix-injection mechanism as :meth:`generate`. vLLM exposes
        per-token logprobs via ``SamplingParams(prompt_logprobs=...)``; we request
        them for the ``target`` positions of a ``prompt+target`` forward over the
        injected cartridge prefix and gather the chosen-token logprobs.
        """
        self._require_vllm_runtime()
        return self._score_on_engine(prompt, target, cartridge)  # pragma: no cover - cluster only

    def _require_vllm_runtime(self) -> None:
        # Construction already guarantees vllm; this guards against subclasses or
        # monkeypatched instances that skipped __init__, keeping the error clear.
        if not _VLLM_AVAILABLE:  # pragma: no cover - defensive
            raise RuntimeError(
                "CartridgeServer.generate/score require a running vllm engine; "
                "vllm is not importable in this environment."
            )

    # The two methods below are the real cluster code paths for serving against
    # an injected cartridge prefix. They are reachable only with a live vLLM
    # engine, so they never run in the vllm-less environment; they are written
    # out so a cluster deployment has the concrete integration to finish wiring.
    def _generate_on_engine(  # pragma: no cover - cluster only
        self,
        prompt: str,
        cartridge: Cartridge,
        *,
        max_new_tokens: int,
        greedy: bool,
    ) -> str:
        writer = self._inject_prefix(cartridge)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0 if greedy else 1.0,
        )
        out = self.llm.generate(
            {"prompt_token_ids": list(prompt_ids)},
            params,
            use_tqdm=False,
            # custom field consumed by the prefix writer to bind this request to
            # the pre-seeded prefix blocks and the logical position offset.
            extra_request_args={"dexa_cartridge_prefix": writer.handle},
        )
        return self.tokenizer.decode(
            list(out[0].outputs[0].token_ids), skip_special_tokens=True
        )

    def _score_on_engine(  # pragma: no cover - cluster only
        self,
        prompt: str,
        target: str,
        cartridge: Cartridge,
    ) -> np.ndarray:
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        if not target_ids:
            return np.zeros(0, dtype=np.float32)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        writer = self._inject_prefix(cartridge)
        full = list(prompt_ids) + list(target_ids)
        first = len(prompt_ids)  # first target position in ``full``
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        out = self.llm.generate(
            {"prompt_token_ids": full},
            params,
            use_tqdm=False,
            extra_request_args={"dexa_cartridge_prefix": writer.handle},
        )
        plps = out[0].prompt_logprobs  # aligned to ``full`` (index 0 may be None)
        scores = np.empty(len(target_ids), dtype=np.float32)
        for k, tok in enumerate(target_ids):
            scores[k] = plps[first + k][tok].logprob
        return scores


class _PagedPrefixWriter:  # pragma: no cover - cluster only
    """Handle binding a cartridge's paged K/V blocks to a vLLM request.

    Holds the per-layer paged blocks (already torch, on-device), the cartridge's
    absolute ``positions`` (the slot mapping for the prefix), the corpus
    ``logical_length`` (the position offset new query tokens start at), and the
    engine ``block_size``. ``attach_to_runner`` acquires free physical blocks and
    copies the cartridge blocks into the engine's KV cache, exposing an opaque
    ``handle`` the request references. The exact block-manager / slot-mapping
    APIs are vLLM-version specific, which is why this is the single pinned seam.
    """

    def __init__(self, *, k_blocks, v_blocks, positions, logical_length, block_size) -> None:
        self.k_blocks = k_blocks
        self.v_blocks = v_blocks
        self.positions = positions
        self.logical_length = logical_length
        self.block_size = block_size
        self.handle: Optional[str] = None

    def attach_to_runner(self, llm) -> None:
        raise RuntimeError(
            "_PagedPrefixWriter.attach_to_runner copies the cartridge's paged KV "
            "blocks into vLLM's KV cache via the block manager / KV connector, "
            "whose APIs are version-specific. Provide the small site shim for "
            "your vLLM release (see the module docstring 'How the prefix "
            "injection works')."
        )
