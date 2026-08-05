"""Tenants and BYOC routing.

A tenant is an API key plus where its requests should be served. Three modes:

  hosted  -> Dexa runs the backend; requests forward to DEXA's managed VLM endpoint.
  byoc    -> the customer runs the backend (and usually this gateway too) in their own
             cloud; requests forward to *their* OpenAI-compatible URL, so screenshots never
             leave their network. Dexa supplies the serving recipe + the savings telemetry.
  mock    -> no backend; synthesizes a response (local demos, CI).

Registry sources, in priority order:
  1. DEXA_TENANTS=<path to json>   — a list of tenant objects (multi-tenant hosted control plane)
  2. env single-tenant            — DEXA_BACKEND_URL / DEXA_API_KEY / DEXA_MODE (the BYOC
                                     self-host case: one team, one key, their own backend)
  3. built-in "dexa-demo" mock tenant so the examples run with zero config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEMO_KEY = "dexa-demo"


@dataclass
class Tenant:
    key: str
    name: str
    mode: str = "hosted"                 # hosted | byoc | mock
    backend_url: str = ""                # OpenAI-compatible backend; empty => mock
    backend_model: str = "dexa-cua-vlm"
    default_baseline: str = "gpt-4o"
    cache_enabled: bool = True           # exact-frame dedup cache (in-memory, per session)

    @property
    def uses_mock(self) -> bool:
        return self.mode == "mock" or not self.backend_url

    def public(self) -> dict:
        return {"name": self.name, "mode": self.mode,
                "backend": ("mock" if self.uses_mock else self.backend_url),
                "backend_model": self.backend_model,
                "default_baseline": self.default_baseline,
                "cache_enabled": self.cache_enabled}


class TenantRegistry:
    def __init__(self, tenants: list[Tenant], *, require_auth: bool) -> None:
        self._by_key = {t.key: t for t in tenants}
        self.require_auth = require_auth

    def resolve(self, key: str | None) -> Tenant | None:
        if key and key in self._by_key:
            return self._by_key[key]
        if not self.require_auth:               # dev convenience: fall back to the demo tenant
            return self._by_key.get(DEMO_KEY)
        return None

    def summary(self) -> dict:
        return {"require_auth": self.require_auth,
                "tenants": {t.name: t.public() for t in self._by_key.values()}}

    # ---- construction ---------------------------------------------------------------------
    @classmethod
    def load(cls) -> "TenantRegistry":
        require_auth = os.environ.get("DEXA_REQUIRE_AUTH") == "1"
        path = os.environ.get("DEXA_TENANTS")
        if path:
            with open(path) as f:
                raw = json.load(f)
            tenants = [cls._from_dict(d) for d in raw]
            if not any(t.key == DEMO_KEY for t in tenants):
                tenants.append(cls._demo())
            return cls(tenants, require_auth=require_auth)

        tenants = [cls._demo()]
        backend = os.environ.get("DEXA_BACKEND_URL", "").rstrip("/")
        if backend:
            tenants.append(Tenant(
                key=os.environ.get("DEXA_API_KEY", DEMO_KEY),
                name=os.environ.get("DEXA_TENANT_NAME", "default"),
                mode=os.environ.get("DEXA_MODE", "hosted"),
                backend_url=backend,
                backend_model=os.environ.get("DEXA_BACKEND_MODEL", "dexa-cua-vlm"),
                default_baseline=os.environ.get("DEXA_BASELINE", "gpt-4o"),
                cache_enabled=os.environ.get("DEXA_CACHE", "1") != "0",
            ))
        return cls(tenants, require_auth=require_auth)

    @staticmethod
    def _demo() -> Tenant:
        # in mock unless an env backend is also configured for it elsewhere
        return Tenant(key=DEMO_KEY, name="demo", mode="mock",
                      backend_url=os.environ.get("DEXA_BACKEND_URL", "").rstrip("/") if
                      os.environ.get("DEXA_DEMO_USES_BACKEND") == "1" else "",
                      cache_enabled=os.environ.get("DEXA_CACHE", "1") != "0")

    @staticmethod
    def _from_dict(d: dict) -> Tenant:
        return Tenant(
            key=d["key"], name=d.get("name", d["key"][:8]),
            mode=d.get("mode", "hosted"),
            backend_url=str(d.get("backend_url", "")).rstrip("/"),
            backend_model=d.get("backend_model", "dexa-cua-vlm"),
            default_baseline=d.get("default_baseline", "gpt-4o"),
            cache_enabled=bool(d.get("cache_enabled", True)),
        )
