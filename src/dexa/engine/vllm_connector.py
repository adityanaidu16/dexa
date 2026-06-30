"""A vLLM **KV-connector** that gives an existing vLLM server persistent,
portable KV state — Dexa's session store wired into vLLM's V1 connector seam.

This is the *zero-code* integration surface. A builder who already runs vLLM
turns it on with one flag at launch (no change to their serving stack)::

    vllm serve <model> \\
        --kv-transfer-config '{"kv_connector":"DexaConnector","kv_connector_module_path":"dexa.engine.vllm_connector","kv_role":"kv_both"}'

After that, the KV blocks vLLM computes for a request are **saved** (keyed by a
content hash of the request's token prefix) through Dexa's
:class:`~dexa.session.store.SessionStore`, and an identical prefix arriving on
*another* instance — or on the same instance after a restart — is **loaded**
back instead of being re-prefilled. That is the same plug-point LMCache uses
(``KVConnectorBase_V1``); Dexa's connector just persists to / restores from the
portable ``.npz`` session state (:func:`dexa.session.state.save_kvcache` /
:func:`~dexa.session.state.load_kvcache`), so the win is cross-request /
cross-instance / post-restart KV reuse.

What runs where
---------------
vLLM (and torch/CUDA) live **only** behind an import guard in this file, exactly
like :mod:`dexa.engine.vllm_backend` and :mod:`dexa.engine.vllm_cartridge`:

* When vLLM is importable, :class:`DexaConnector` subclasses vLLM's real
  ``KVConnectorBase_V1`` and the lifecycle hooks run natively in the engine.
* When vLLM is absent (laptop / CI / this Mac), the module still imports: a
  stand-in base is defined so the class and its methods stay introspectable, and
  the pure-numpy / persistence helpers are fully unit-testable. Constructing
  :class:`DexaConnector` without vLLM raises a helpful ``RuntimeError`` (the same
  choice :class:`~dexa.engine.vllm_backend.VLLMBackend` makes), and
  ``engine/__init__`` does not import this module (use
  ``import dexa.engine.vllm_connector``), keeping vLLM optional.

The block<->numpy movement is delegated to small **pure-numpy** helpers
(:func:`prefix_key`, :func:`kvcache_to_paged_blocks`,
:func:`paged_blocks_to_kvcache`) that need no GPU and are tested everywhere; the
engine-touching glue (writing into / reading from vLLM's paged KV tensors) is the
version-pinned seam, marked ``pragma: no cover`` and raising a clear error if
reached without vLLM.

The V1 connector lifecycle (what the methods do)
------------------------------------------------
vLLM drives a connector from two sides (mirroring the engine-agnostic lifecycle
in :mod:`dexa.serving.session_manager`, but over vLLM's *paged* KV):

* **Scheduler side** (one connector instance, ``KVConnectorRole.SCHEDULER``):
    - :meth:`get_num_new_matched_tokens` — on a new request, hash its prompt
      prefix and ask the store "do we already have this?"; report how many
      tokens can be loaded externally (so the scheduler skips re-prefilling
      them).
    - :meth:`update_state_after_alloc` — record the KV blocks the scheduler
      allocated for those external tokens (where the worker must write them).
    - :meth:`build_connector_meta` — pack the per-request load/save plan into the
      metadata object handed to the worker for the step.
    - :meth:`request_finished` — when a request ends, decide whether its KV is
      worth persisting and key it for save.

* **Worker side** (per-worker instance, ``KVConnectorRole.WORKER``):
    - :meth:`register_kv_caches` — capture handles to the engine's paged KV
      tensors (per layer).
    - :meth:`start_load_kv` / :meth:`wait_for_layer_load` — for each request
      marked for load, restore the KVCache from the store, convert it to paged
      blocks, and copy it into the allocated blocks (layer-synchronized).
    - :meth:`save_kv_layer` / :meth:`wait_for_save` — collect each layer's KV
      blocks for requests marked for save, assemble a :class:`KVCache`, and
      persist it through the store.
    - :meth:`get_finished` — report which async load/save transfers have
      completed.

Version caveat
--------------
vLLM's V1 KV-connector API (method names, signatures, and the
``KVConnectorMetadata`` shape) is **vLLM-version-specific** and still evolving.
The signatures below target the documented ``KVConnectorBase_V1`` surface (the
scheduler/worker split used by LMCache). vLLM is **not importable in this build
environment**, so this was coded against the *documented* V1 interface rather
than a pinned release — validate the exact signatures against the
``vllm.distributed.kv_transfer.kv_connector.v1.base`` module on your real vLLM
before relying on the in-engine paths. The pure-numpy helpers are stable and are
tested here regardless.

Run the structural tests anywhere::

    .venv/bin/python -m pytest tests/test_vllm_connector.py -v
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dexa.core.types import KVCache, LayerKV, ModelSpec, hash_tokens
from dexa.engine.vllm_cartridge import pack_token_major_into_blocks
from dexa.session.state import load_kvcache, save_kvcache  # noqa: F401  (used on cluster)
from dexa.session.store import SessionStore

# Import-guard vLLM so this module imports on machines without it (laptop, CI).
# When present we subclass the real V1 connector base; when absent we fall back
# to lightweight stand-ins so the module — and ``DexaConnector`` — stay
# importable and introspectable. All vLLM/torch usage happens only after a
# successful construction, which itself errors if vLLM is missing.
try:  # pragma: no cover - exercised only on the cluster
    import vllm  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    _VLLM_AVAILABLE = True
    _VLLM_IMPORT_ERROR: Optional[BaseException] = None
    _VLLM_VERSION: Optional[str] = getattr(vllm, "__version__", None)
except Exception as exc:  # ImportError, or CUDA/driver/layout drift on import
    vllm = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False
    _VLLM_IMPORT_ERROR = exc
    _VLLM_VERSION = None

    class KVConnectorRole:  # type: ignore[no-redef]
        """Stand-in for ``vllm...KVConnectorRole`` so the role constants exist
        when vLLM is absent (values mirror the real enum's two roles)."""

        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorMetadata:  # type: ignore[no-redef]
        """Stand-in for the per-step metadata object vLLM passes worker-side."""

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        """Minimal stand-in for vLLM's ``KVConnectorBase_V1`` ABC.

        Provides only the connector-metadata plumbing the real base offers
        (``bind``/``clear``/``_get_connector_metadata``) so :class:`DexaConnector`
        is structurally identical with or without vLLM. Every engine-touching
        hook is overridden by :class:`DexaConnector`.
        """

        def __init__(self, vllm_config: Any, role: Any) -> None:
            self._vllm_config = vllm_config
            self._role = role
            self._connector_metadata: Any = None

        def bind_connector_metadata(self, connector_metadata: Any) -> None:
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self) -> Any:
            return self._connector_metadata


def vllm_available() -> bool:
    """Whether vLLM imported successfully in this process."""
    return _VLLM_AVAILABLE


def vllm_version() -> Optional[str]:
    """The vLLM version this process imported (``None`` if vLLM is absent).

    Surfaced so a site can confirm which V1 connector interface the in-engine
    paths were validated against.
    """
    return _VLLM_VERSION


# ---------------------------------------------------------------------------
# Pure-numpy helpers (no vLLM): unit-tested everywhere.
# ---------------------------------------------------------------------------
def prefix_key(token_ids, *, model_name: Optional[str] = None) -> str:
    """Content-hash key for a request's token prefix.

    This is the identity under which a prefix's KV is stored and looked up, so an
    identical prefix on another instance / after a restart maps to the same key
    and loads instead of re-prefilling. Deterministic and collision-resistant:
    built on :func:`dexa.core.types.hash_tokens` (a 128-bit blake2b over the
    length-prefixed token ids, so order *and* length change the digest). The
    optional ``model_name`` is folded in (sanitized to a filesystem-safe string)
    so caches for different models never collide in a shared store.

    Returns a string safe to use as a :class:`SessionStore` id (a filename stem).
    """
    digest = hash_tokens(list(token_ids))
    if model_name:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in model_name)
        return f"{safe}-{digest}"
    return f"dexa-{digest}"


def kvcache_to_paged_blocks(
    kv: KVCache, block_size: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Convert a :class:`KVCache` to vLLM-style paged KV blocks, per layer.

    Each ``LayerKV`` (``key``/``value`` are ``[n_kv_heads, T, head_dim]``) is
    moved to token-major ``[T, n_kv_heads, head_dim]`` and packed into fixed-size
    blocks ``[num_blocks, block_size, n_kv_heads, head_dim]`` with the final
    partial block zero-padded (slot ``i`` -> block ``i // block_size``, offset
    ``i % block_size``) — exactly how vLLM lays a contiguous sequence into KV
    blocks. Pure numpy; the cluster path copies these blocks into the engine's
    paged KV tensors. Round-trips with :func:`paged_blocks_to_kvcache`.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    k_blocks: list[np.ndarray] = []
    v_blocks: list[np.ndarray] = []
    for layer in kv.layers:
        key = np.asarray(layer.key)
        val = np.asarray(layer.value)
        if key.ndim != 3 or val.ndim != 3:
            raise ValueError(
                f"expected per-layer [n_kv_heads, T, head_dim]; got {key.shape} / {val.shape}"
            )
        km = np.ascontiguousarray(key.transpose(1, 0, 2))   # [T, n_kv, d]
        vm = np.ascontiguousarray(val.transpose(1, 0, 2))
        k_blocks.append(pack_token_major_into_blocks(km, block_size))
        v_blocks.append(pack_token_major_into_blocks(vm, block_size))
    return k_blocks, v_blocks


def paged_blocks_to_kvcache(
    k_blocks: list[np.ndarray],
    v_blocks: list[np.ndarray],
    *,
    spec: ModelSpec,
    positions: np.ndarray,
    token_ids: Optional[list[int]] = None,
    meta: Optional[dict] = None,
) -> KVCache:
    """Inverse of :func:`kvcache_to_paged_blocks`.

    Takes per-layer paged blocks ``[num_blocks, block_size, n_kv_heads, head_dim]``
    and the ``positions`` (length ``T``) that index the real tokens, drops the
    block padding, and rebuilds a :class:`KVCache` in storage layout
    (``[n_kv_heads, T, head_dim]`` per layer). This is the worker-side read path:
    gather a sequence's KV blocks out of the engine and turn them back into the
    portable numpy cache Dexa persists.
    """
    if len(k_blocks) != len(v_blocks):
        raise ValueError("k_blocks/v_blocks length mismatch")
    positions = np.asarray(positions, dtype=np.int64)
    T = int(positions.shape[0])
    layers: list[LayerKV] = []
    for kb, vb in zip(k_blocks, v_blocks):
        kb = np.asarray(kb)
        vb = np.asarray(vb)
        if kb.ndim != 4 or vb.ndim != 4:
            raise ValueError(
                f"expected paged blocks [num_blocks, block_size, n_kv, d]; got {kb.shape}"
            )
        nb, bs, n_kv, d = kb.shape
        km = kb.reshape(nb * bs, n_kv, d)[:T]   # [T, n_kv, d]
        vm = vb.reshape(nb * bs, n_kv, d)[:T]
        layers.append(
            LayerKV(
                key=np.ascontiguousarray(km.transpose(1, 0, 2)),     # [n_kv, T, d]
                value=np.ascontiguousarray(vm.transpose(1, 0, 2)),
            )
        )
    return KVCache(
        spec=spec,
        layers=layers,
        positions=positions,
        token_ids=list(token_ids) if token_ids is not None else None,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# The connector.
# ---------------------------------------------------------------------------
class DexaConnector(KVConnectorBase_V1):
    """Persist/restore vLLM's paged KV through Dexa's :class:`SessionStore`.

    Subclasses vLLM's ``KVConnectorBase_V1`` when vLLM is present; otherwise the
    stand-in base above keeps the class importable and introspectable. The
    block<->numpy movement is delegated to the module-level pure-numpy helpers,
    and the store round-trip to the :meth:`store_kvcache` / :meth:`load_kvcache_for`
    static helpers — all testable without vLLM. The methods that touch the
    engine's paged KV tensors are ``pragma: no cover`` and raise a clear
    ``RuntimeError`` if reached without vLLM.

    Onboarding (no change to the builder's serving stack)::

        vllm serve <model> --kv-transfer-config \\
          '{"kv_connector":"DexaConnector","kv_connector_module_path":"dexa.engine.vllm_connector","kv_role":"kv_both"}'

    The store root can be set via the connector's extra config, e.g.
    ``--kv-transfer-config '{..., "kv_connector_extra_config":{"dexa_store_root":"/mnt/kv"}}'``.
    """

    #: default on-disk root for persisted session KV (overridable via extra config).
    DEFAULT_STORE_ROOT = ".dexa_kv_connector"

    def __init__(self, vllm_config: Any = None, role: Any = None) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "DexaConnector requires the 'vllm' package, which is not importable "
                "in this environment. It is loaded by vLLM itself when you launch "
                "with `--kv-transfer-config '{\"kv_connector\":\"DexaConnector\","
                "\"kv_connector_module_path\":\"dexa.engine.vllm_connector\","
                "\"kv_role\":\"kv_both\"}'` on a host with `pip install 'vllm>=0.6'`. "
                f"Original import error: {_VLLM_IMPORT_ERROR!r}. The pure-numpy "
                "helpers (prefix_key, kvcache_to_paged_blocks, paged_blocks_to_kvcache) "
                "and the store helpers (store_kvcache, load_kvcache_for) work "
                "everywhere without vllm."
            )
        self._init_runtime(vllm_config, role)  # pragma: no cover - cluster only

    def _init_runtime(self, vllm_config: Any, role: Any) -> None:  # pragma: no cover - cluster only
        """Real construction body (cluster only): wire the store, the model spec,
        and the per-request bookkeeping vLLM's V1 lifecycle needs."""
        super().__init__(vllm_config, role)
        self._role = role
        # Store root from the connector's extra config, if provided.
        root = self.DEFAULT_STORE_ROOT
        extra = self._extra_config(vllm_config)
        if extra:
            root = extra.get("dexa_store_root", root)
        self.store = SessionStore(root=root)
        self.model_name = self._model_name(vllm_config)
        # paged KV tensors per layer, captured on the worker via register_kv_caches.
        self._kv_caches: dict[str, Any] = {}
        # scheduler-side: requests that matched the store this step (req_id -> key),
        # and the blocks allocated to receive their loaded KV (req_id -> block_ids).
        self._needs_load: dict[str, str] = {}
        self._load_blocks: dict[str, list[int]] = {}
        # requests whose KV should be persisted on finish (req_id -> (key, block_ids)).
        self._needs_save: dict[str, tuple[str, list[int]]] = {}
        # accumulator while save_kv_layer streams layers for the current step.
        self._save_layers: dict[str, dict[str, np.ndarray]] = {}

    # --- config introspection (cluster only) ------------------------------
    @staticmethod
    def _extra_config(vllm_config: Any) -> dict:  # pragma: no cover - cluster only
        kt = getattr(vllm_config, "kv_transfer_config", None)
        if kt is None:
            return {}
        return getattr(kt, "kv_connector_extra_config", None) or {}

    @staticmethod
    def _model_name(vllm_config: Any) -> str:  # pragma: no cover - cluster only
        mc = getattr(vllm_config, "model_config", None)
        return getattr(mc, "model", "") if mc is not None else ""

    def _require_vllm(self) -> None:
        if not _VLLM_AVAILABLE:  # pragma: no cover - defensive; __init__ already guards
            raise RuntimeError(
                "DexaConnector engine hooks require a running vLLM; vllm is not "
                "importable in this environment."
            )

    # --- store helpers (pure python; testable without vLLM) ---------------
    @staticmethod
    def store_kvcache(
        store: SessionStore, token_ids, kv: KVCache, *, model_name: Optional[str] = None
    ) -> str:
        """Persist ``kv`` under the content-hash key of ``token_ids`` and return
        the key. The persistence tier through which cross-instance / post-restart
        reuse happens; pure Dexa (no vLLM)."""
        key = prefix_key(token_ids, model_name=model_name)
        store.save(key, kv)
        return key

    @staticmethod
    def load_kvcache_for(
        store: SessionStore, token_ids, *, model_name: Optional[str] = None
    ) -> Optional[KVCache]:
        """Restore the KVCache for ``token_ids`` if the store has it, else
        ``None``. Pure Dexa (no vLLM)."""
        key = prefix_key(token_ids, model_name=model_name)
        if not store.has(key):
            return None
        kv, _ = store.load(key)
        return kv

    @staticmethod
    def has_prefix(
        store: SessionStore, token_ids, *, model_name: Optional[str] = None
    ) -> bool:
        """Whether the store already holds KV for this token prefix."""
        return store.has(prefix_key(token_ids, model_name=model_name))

    # --- request token access ---------------------------------------------
    @staticmethod
    def _request_tokens(request: Any) -> list[int]:  # pragma: no cover - cluster only
        """The prompt token ids of a vLLM request (attribute name drifts across
        releases; probe the known spellings)."""
        for attr in ("prompt_token_ids", "all_token_ids", "token_ids"):
            toks = getattr(request, attr, None)
            if toks is not None:
                return list(toks)
        raise RuntimeError("could not read token ids off the vLLM request object")

    # =====================================================================
    # Scheduler-side V1 hooks (cluster only).
    # =====================================================================
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:  # pragma: no cover - cluster only
        """How many tokens of ``request`` can be served from external KV.

        Hash the request's token prefix and ask the store. If we hold KV for it,
        report the count of tokens beyond what vLLM has already computed
        (``num_computed_tokens``) so the scheduler skips re-prefilling them, and
        remember the key so :meth:`build_connector_meta` schedules the load. The
        second tuple element is the async flag (``False`` => the load is
        synchronous within the step). Returns ``(0, False)`` on a miss.
        """
        self._require_vllm()
        tokens = self._request_tokens(request)
        key = prefix_key(tokens, model_name=self.model_name)
        if not self.store.has(key):
            return 0, False
        n_external = max(0, len(tokens) - int(num_computed_tokens))
        if n_external == 0:
            return 0, False
        self._needs_load[self._req_id(request)] = key
        return n_external, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:  # pragma: no cover - cluster only
        """Record the KV blocks the scheduler allocated to receive the loaded KV.

        Called after :meth:`get_num_new_matched_tokens` reported a hit; ``blocks``
        describes the physical blocks the worker must fill from the store. We keep
        their ids keyed by request so :meth:`build_connector_meta` can hand the
        worker an exact write plan.
        """
        self._require_vllm()
        req_id = self._req_id(request)
        if req_id in self._needs_load and num_external_tokens > 0:
            self._load_blocks[req_id] = self._block_ids(blocks)

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:  # pragma: no cover - cluster only
        """Package this step's per-request load/save plan for the worker.

        Drains the scheduler-side maps (loads matched this step; saves queued by
        :meth:`request_finished`) into a :class:`_DexaConnectorMetadata` the
        engine binds on the worker before :meth:`start_load_kv` /
        :meth:`save_kv_layer` run.
        """
        self._require_vllm()
        meta = _DexaConnectorMetadata(
            loads={rid: (key, self._load_blocks.get(rid, [])) for rid, key in self._needs_load.items()},
            saves=dict(self._needs_save),
        )
        self._needs_load.clear()
        self._load_blocks.clear()
        self._needs_save.clear()
        return meta

    def request_finished(
        self, request: Any, block_ids: Any
    ) -> tuple[bool, Optional[dict[str, Any]]]:  # pragma: no cover - cluster only
        """On request completion, queue its KV for persistence.

        Keys the finished sequence by its full token prefix and records the blocks
        holding its KV so :meth:`save_kv_layer` / :meth:`wait_for_save` can read
        them out and persist a :class:`KVCache`. Returns ``(False, None)``: the
        blocks may be freed immediately (we copy out synchronously), and there is
        no async transfer state to hand back to the scheduler.
        """
        self._require_vllm()
        tokens = self._request_tokens(request)
        key = prefix_key(tokens, model_name=self.model_name)
        if not self.store.has(key):
            self._needs_save[self._req_id(request)] = (key, self._block_ids(block_ids))
        return False, None

    # =====================================================================
    # Worker-side V1 hooks (cluster only).
    # =====================================================================
    def register_kv_caches(self, kv_caches: dict) -> None:  # pragma: no cover - cluster only
        """Capture handles to the engine's paged KV tensors (one per layer), the
        targets/sources for the block copies in :meth:`start_load_kv` /
        :meth:`save_kv_layer`."""
        self._require_vllm()
        self._kv_caches = dict(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:  # pragma: no cover - cluster only
        """Restore matched prefixes into their allocated blocks.

        For each request the bound metadata marks for load: read the KVCache from
        the store, convert it to paged blocks with :func:`kvcache_to_paged_blocks`
        (using the engine ``block_size``), move to torch on the layer tensors'
        device/dtype, and scatter the blocks into the allocated block ids of each
        layer's paged KV. After this, the scheduler-skipped prefix tokens already
        have correct KV, so decode proceeds with no re-prefill.
        """
        self._require_vllm()
        import torch  # vllm pulls in torch; safe on the cluster

        meta: _DexaConnectorMetadata = self._get_connector_metadata()
        if meta is None or not meta.loads:
            return
        block_size = self._block_size()
        for _req_id, (key, block_ids) in meta.loads.items():
            kv, _ = self.store.load(key)
            k_blocks, v_blocks = kvcache_to_paged_blocks(kv, block_size)
            for li, layer_name in enumerate(self._kv_caches):
                k_dst, v_dst = self._layer_kv_tensors(layer_name)
                k_src = torch.from_numpy(k_blocks[li]).to(k_dst.device, k_dst.dtype)
                v_src = torch.from_numpy(v_blocks[li]).to(v_dst.device, v_dst.dtype)
                for bi, phys in enumerate(block_ids[: k_src.shape[0]]):
                    k_dst[phys] = k_src[bi]
                    v_dst[phys] = v_src[bi]

    def wait_for_layer_load(self, layer_name: str) -> None:  # pragma: no cover - cluster only
        """Block until ``layer_name``'s load finished. The store path copies
        synchronously inside :meth:`start_load_kv`, so this is a barrier no-op;
        an async/streaming store would join the per-layer transfer here."""
        self._require_vllm()

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any
    ) -> None:  # pragma: no cover - cluster only
        """Collect one layer's KV for requests queued for save.

        vLLM streams each layer's freshly-computed KV through this hook. For every
        request the bound metadata marks for save, gather its blocks for this
        layer (numpy) and stash them; :meth:`wait_for_save` assembles the layers
        into a :class:`KVCache` and persists it.
        """
        self._require_vllm()
        meta: _DexaConnectorMetadata = self._get_connector_metadata()
        if meta is None or not meta.saves:
            return
        k_cache, v_cache = self._split_kv_layer(kv_layer)
        for req_id, (_key, block_ids) in meta.saves.items():
            k_blocks = np.stack([self._block_to_numpy(k_cache[b]) for b in block_ids], axis=0)
            v_blocks = np.stack([self._block_to_numpy(v_cache[b]) for b in block_ids], axis=0)
            slot = self._save_layers.setdefault(req_id, {"k": [], "v": []})
            slot["k"].append(k_blocks)
            slot["v"].append(v_blocks)

    def wait_for_save(self) -> None:  # pragma: no cover - cluster only
        """Persist the KV collected across :meth:`save_kv_layer` calls.

        For each saved request, rebuild a :class:`KVCache` from the per-layer
        paged blocks (:func:`paged_blocks_to_kvcache`) and write it to the store
        under its prefix key. After this returns, an identical prefix on any
        instance loads instead of re-prefilling.
        """
        self._require_vllm()
        meta: _DexaConnectorMetadata = self._get_connector_metadata()
        if meta is None or not self._save_layers:
            return
        spec = self._spec()
        for req_id, layers in self._save_layers.items():
            key, _block_ids = meta.saves[req_id]
            k_blocks = layers["k"]
            v_blocks = layers["v"]
            T, positions, token_ids = self._save_geometry(req_id, meta)
            kv = paged_blocks_to_kvcache(
                k_blocks, v_blocks, spec=spec, positions=positions, token_ids=token_ids
            )
            self.store.save(key, kv)
        self._save_layers.clear()

    def get_finished(
        self, finished_req_ids: set
    ) -> tuple[Optional[set], Optional[set]]:  # pragma: no cover - cluster only
        """Report which async saves/loads completed this step. The store path is
        synchronous, so nothing is pending: return ``(None, None)``."""
        self._require_vllm()
        return None, None

    # --- version-pinned engine glue (cluster only) ------------------------
    # The helpers below reach into vLLM-version-specific internals (request ids,
    # block descriptors, paged-tensor layout, model spec). They are isolated here
    # so a site pins exactly one place to its vLLM release. Each raises a clear
    # error if reached without the engine wiring, mirroring vllm_cartridge's
    # _PagedPrefixWriter seam.
    @staticmethod
    def _req_id(request: Any) -> str:  # pragma: no cover - cluster only
        for attr in ("request_id", "req_id", "request_id_str"):
            rid = getattr(request, attr, None)
            if rid is not None:
                return str(rid)
        raise RuntimeError("could not read request id off the vLLM request object")

    @staticmethod
    def _block_ids(blocks: Any) -> list[int]:  # pragma: no cover - cluster only
        raise RuntimeError(
            "DexaConnector._block_ids extracts physical block ids from the vLLM "
            "scheduler's block descriptor, whose shape is version-specific. "
            "Provide the small site shim for your vLLM release (see the module "
            "docstring 'Version caveat')."
        )

    def _block_size(self) -> int:  # pragma: no cover - cluster only
        return self._vllm_config.cache_config.block_size

    def _spec(self) -> ModelSpec:  # pragma: no cover - cluster only
        raise RuntimeError(
            "DexaConnector._spec builds a ModelSpec from vLLM's model_config; the "
            "exact config fields are version-specific (see vllm_backend.__init__ "
            "for the field probing). Provide the site shim for your vLLM release."
        )

    def _layer_kv_tensors(self, layer_name: str):  # pragma: no cover - cluster only
        raise RuntimeError(
            "DexaConnector._layer_kv_tensors splits the registered paged KV tensor "
            "for a layer into (key_cache, value_cache); the packed layout "
            "(separate K/V, stacked, or MLA) is version- and backend-specific. "
            "Provide the site shim for your vLLM release."
        )

    def _split_kv_layer(self, kv_layer: Any):  # pragma: no cover - cluster only
        raise RuntimeError(
            "DexaConnector._split_kv_layer splits the per-layer KV tensor handed "
            "to save_kv_layer into (key_cache, value_cache); layout is "
            "version-/backend-specific. Provide the site shim for your vLLM release."
        )

    @staticmethod
    def _block_to_numpy(block: Any) -> np.ndarray:  # pragma: no cover - cluster only
        return block.to("cpu").float().numpy()

    def _save_geometry(self, req_id: str, meta: Any):  # pragma: no cover - cluster only
        raise RuntimeError(
            "DexaConnector._save_geometry recovers (T, positions, token_ids) for a "
            "finished request from the scheduler/connector metadata; the carrier "
            "for the token ids and sequence length is version-specific. Provide "
            "the site shim for your vLLM release."
        )


class _DexaConnectorMetadata(KVConnectorMetadata):  # pragma: no cover - cluster only
    """Per-step worker metadata: the load and save plans the scheduler-side
    connector built, bound on the worker before its hooks run.

    ``loads``: ``req_id -> (store_key, block_ids)`` to restore into.
    ``saves``: ``req_id -> (store_key, block_ids)`` to read out and persist.
    """

    def __init__(self, *, loads: dict, saves: dict) -> None:
        super().__init__()
        self.loads = loads
        self.saves = saves
