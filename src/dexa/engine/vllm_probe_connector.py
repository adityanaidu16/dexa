"""Diagnostic vLLM V1 KV-connector that RECORDS the structure of every object the
real lifecycle hands its hooks — without moving any KV.

Purpose. :class:`~dexa.engine.vllm_connector.DexaConnector` has five version-pinned
site-shims (`_block_ids`, `_spec`, `_layer_kv_tensors`, `_split_kv_layer`,
`_save_geometry`) that deliberately raise because the exact shapes are
vLLM-version-specific and unknown without a running engine. This probe is how you
discover them: load it into a real vLLM on a GPU box, run one generation, and it
dumps — as JSON — the concrete type/shape/attributes of:

* the registered paged KV tensors per layer (for `_layer_kv_tensors`/`_split_kv_layer`),
* the ``blocks`` descriptor the scheduler allocates (for `_block_ids`),
* the ``request`` object (token ids, request id, geometry — for `_save_geometry`),
* the ``vllm_config.model_config`` / ``cache_config`` (for `_spec`, ``block_size``),
* the ``scheduler_output`` and the per-layer ``kv_layer`` handed to ``save_kv_layer``.

It never matches or transfers KV (every hook returns the safe "nothing to do"
value), so generation runs to completion normally while the structure is captured.

Run on a GPU box (see ``scripts/modal_connector_probe.py``)::

    vllm serve facebook/opt-125m --enforce-eager --kv-transfer-config \\
      '{"kv_connector":"DexaProbeConnector","kv_connector_module_path":"dexa.engine.vllm_probe_connector","kv_role":"kv_both","kv_connector_extra_config":{"dexa_probe_out":"/tmp/probe.json"}}'

or via the offline ``LLM(...)`` API. The dump goes to ``dexa_probe_out`` (default
``/tmp/dexa_probe.json``) and is also printed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

try:  # pragma: no cover - exercised only on the cluster
    import vllm  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,  # noqa: F401
    )

    _VLLM_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _VLLM_AVAILABLE = False
    _IMPORT_ERR = exc

    class KVConnectorMetadata:  # type: ignore[no-redef]
        pass

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        def __init__(self, vllm_config: Any, role: Any) -> None:
            self._vllm_config = vllm_config
            self._role = role


def describe(obj: Any, *, depth: int = 3, _seen: Optional[set] = None) -> Any:
    """A JSON-safe structural summary of ``obj``: type, and shape/dtype for tensors,
    element structure for containers, and a filtered attribute map for objects.
    Bounded in depth and breadth so a dump stays readable."""
    _seen = _seen if _seen is not None else set()
    t = type(obj)
    tn = f"{t.__module__}.{t.__qualname__}"

    # scalars
    if obj is None or isinstance(obj, (bool, int, float, str)):
        s = obj if not isinstance(obj, str) else (obj[:120] + "…" if len(obj) > 120 else obj)
        return s
    # torch tensors / numpy arrays (duck-typed to avoid importing torch here)
    shape = getattr(obj, "shape", None)
    if shape is not None and hasattr(obj, "dtype"):
        return {"__type__": tn, "shape": list(shape),
                "dtype": str(getattr(obj, "dtype", None)),
                "device": str(getattr(obj, "device", "")) or None}
    if depth <= 0:
        return {"__type__": tn, "__truncated__": True}
    # containers
    if isinstance(obj, (list, tuple)):
        head = [describe(x, depth=depth - 1, _seen=_seen) for x in list(obj)[:3]]
        return {"__type__": tn, "len": len(obj), "sample": head}
    if isinstance(obj, dict):
        keys = list(obj.keys())
        sample = {str(k): describe(obj[k], depth=depth - 1, _seen=_seen) for k in keys[:5]}
        return {"__type__": tn, "len": len(obj), "keys_sample": [str(k) for k in keys[:12]],
                "value_sample": sample}
    # avoid cycles
    oid = id(obj)
    if oid in _seen:
        return {"__type__": tn, "__cycle__": True}
    _seen.add(oid)
    # generic object: dump non-callable, non-dunder attributes
    attrs: dict[str, Any] = {}
    names = [n for n in dir(obj) if not n.startswith("_")]
    for n in names[:40]:
        try:
            v = getattr(obj, n)
        except Exception as e:  # attribute access can raise
            attrs[n] = f"<err {type(e).__name__}>"
            continue
        if callable(v):
            continue
        attrs[n] = describe(v, depth=depth - 1, _seen=_seen)
    return {"__type__": tn, "attrs": attrs}


class _ProbeMeta(KVConnectorMetadata):  # pragma: no cover - cluster only
    pass


class DexaProbeConnector(KVConnectorBase_V1):  # pragma: no cover - cluster only
    """Records the structure of the objects each V1 hook receives, then no-ops."""

    _RECORD: dict[str, Any] = {}

    def __init__(self, vllm_config: Any = None, role: Any = None,
                 kv_cache_config: Any = None) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError("DexaProbeConnector requires vllm; import failed: "
                               f"{_IMPORT_ERR!r}")
        # vLLM >=0.24 validates the connector ctor: external V1 connectors MUST take
        # kv_cache_config as the 3rd arg and pass it to super().__init__().
        super().__init__(vllm_config, role, kv_cache_config)
        self._role = role
        kt = getattr(vllm_config, "kv_transfer_config", None)
        extra = getattr(kt, "kv_connector_extra_config", None) or {}
        base = extra.get("dexa_probe_out", "/tmp/dexa_probe.json")
        # vLLM builds separate connector instances (scheduler vs worker), often in
        # separate processes — write per-(role,pid) files so they don't clobber; the
        # runner merges every dexa_probe.*.json.
        role_name = getattr(role, "name", str(role)).lower()
        self._out = base.replace(".json", f".{role_name}.{os.getpid()}.json")
        self._rec("vllm_version", getattr(vllm, "__version__", None))
        self._rec("role", describe(role))
        # _spec / block_size sources:
        self._rec("model_config", describe(getattr(vllm_config, "model_config", None)))
        self._rec("cache_config", describe(getattr(vllm_config, "cache_config", None)))
        self._rec("parallel_config", describe(getattr(vllm_config, "parallel_config", None)))
        # the KV layout the shims need, handed straight to the ctor in vLLM >=0.24.
        self._rec("kv_cache_config", describe(kv_cache_config, depth=5))

    # --- record helper -----------------------------------------------------
    def _rec(self, key: str, val: Any, *, once: bool = True) -> None:
        if once and key in DexaProbeConnector._RECORD:
            return
        DexaProbeConnector._RECORD[key] = val
        try:
            with open(self._out, "w") as f:
                json.dump(DexaProbeConnector._RECORD, f, indent=2, default=str)
        except Exception:
            pass
        print(f"[dexa-probe] recorded {key}", flush=True)

    # --- scheduler side ----------------------------------------------------
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._rec("request", describe(request))
        self._rec("num_computed_tokens", describe(num_computed_tokens))
        return 0, False  # never match -> vLLM does a normal prefill

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._rec("alloc_blocks", describe(blocks))
        self._rec("num_external_tokens", describe(num_external_tokens))

    def build_connector_meta(self, scheduler_output):
        self._rec("scheduler_output", describe(scheduler_output))
        return _ProbeMeta()

    def request_finished(self, request, block_ids):
        self._rec("finished_request", describe(request))
        self._rec("finished_block_ids", describe(block_ids))
        return False, None

    # --- worker side -------------------------------------------------------
    def register_kv_caches(self, kv_caches):
        self._rec("kv_caches", describe(kv_caches, depth=4))

    def start_load_kv(self, forward_context, **kwargs):
        self._rec("forward_context", describe(forward_context))

    def wait_for_layer_load(self, layer_name):
        return None

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        self._rec("save_layer_name", describe(layer_name))
        self._rec("save_kv_layer", describe(kv_layer, depth=4))
        self._rec("save_attn_metadata", describe(attn_metadata))

    def wait_for_save(self):
        return None

    def get_finished(self, finished_req_ids):
        return None, None
