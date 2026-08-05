"""Hot-path key resolver with a short-TTL cache.

The gateway must not hit Postgres on every request. This resolves a presented key to a
principal (org config + ids) from an in-process TTL cache; only cache misses touch the DB.
A revoke/rotate therefore propagates within TTL seconds — the deliberate trade from the
architecture doc (fast, resilient auth vs instant revocation). Swap the dict for Redis to
share the cache across gateway replicas; the interface is the same.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from sqlalchemy import select

from . import db
from . import keys as keymod
from .models import ApiKey, Org

TTL_SECONDS = float(os.environ.get("DEXA_KEY_CACHE_TTL", "30"))
HOSTED_POOL_URL = os.environ.get("DEXA_BACKEND_URL", "").rstrip("/")


@dataclass
class Tenant:
    """Duck-compatible with gateway.tenants.Tenant (what app.py routes on)."""
    name: str
    mode: str
    backend_url: str
    backend_model: str
    default_baseline: str
    cache_enabled: bool

    @property
    def uses_mock(self) -> bool:
        return self.mode == "mock" or not self.backend_url


@dataclass
class Principal:
    tenant: Tenant
    org_id: str
    key_id: str


def _build(org: Org, key_id: str) -> Principal:
    # hosted orgs with no backend of their own use the shared hosted pool
    backend = org.backend_url or (HOSTED_POOL_URL if org.mode == "hosted" else "")
    mode = org.mode if backend else "mock"
    return Principal(
        tenant=Tenant(name=org.name or org.id, mode=mode, backend_url=backend,
                      backend_model=org.backend_model, default_baseline=org.default_baseline,
                      cache_enabled=org.cache_enabled),
        org_id=org.id, key_id=key_id)


class KeyResolver:
    def __init__(self, ttl: float = TTL_SECONDS) -> None:
        self.ttl = ttl
        self._cache: dict[str, tuple[float, Principal | None]] = {}

    def resolve(self, secret: str | None) -> Principal | None:
        if not keymod.looks_like_key(secret):
            return None
        h = keymod.hash_secret(secret)
        hit = self._cache.get(h)
        now = time.monotonic()
        if hit and hit[0] > now:
            return hit[1]
        principal = self._lookup(h)
        self._cache[h] = (now + self.ttl, principal)
        return principal

    def _lookup(self, key_hash: str) -> Principal | None:
        with db.session() as s:
            rec = s.execute(
                select(ApiKey).where(ApiKey.key_hash == key_hash)).scalar_one_or_none()
            if rec is None or rec.revoked_at is not None:
                return None
            org = s.get(Org, rec.org_id)
            if org is None:
                return None
            return _build(org, rec.id)

    def invalidate(self, secret: str | None = None) -> None:
        if secret:
            self._cache.pop(keymod.hash_secret(secret), None)
        else:
            self._cache.clear()
