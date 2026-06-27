"""Tiered KV cache store with prefix-reuse indexing (the LMCache axis).

This is the storage substrate for the *reuse* school of long-context serving
(LMCache and friends): keep raw KV around so that a later request which shares a
prefix can **load** that prefix instead of recomputing it, and spill what does
not fit in fast memory down a tier hierarchy (GPU HBM -> CPU DRAM -> NVMe)
rather than discarding it. Crucially this school **never compacts** -- bytes are
moved, not shrunk -- so retained memory grows with the amount of *unique*
context the system has ever seen. That is exactly the axis we want to contrast
with Dexa's bounded, compacted :class:`~dexa.memory.WorkingMemory`.

:class:`TieredCacheStore` implements the :class:`~dexa.core.types.CacheStore`
protocol. It is deliberately torch-free and in-memory: every cached object only
needs an ``nbytes()`` accountant (both :class:`~dexa.core.types.KVCache` and
:class:`~dexa.core.types.CompactCache` qualify), so the whole tiering/eviction/
reuse story is exercised on CPU with numpy. The NVMe tier is *simulated*: bytes
live in RAM but reads/writes are charged a modeled access latency, so the cost
gradient across tiers is visible in :meth:`stats` without touching disk.

Tiering policy
--------------
* New entries land in the top (``gpu``) tier.
* When a tier exceeds its capacity, its least-recently-used entry is **demoted**
  to the next tier down; an entry demoted off the bottom tier is **dropped**
  (true eviction -- a subsequent reuse lookup misses and forces recompute).
* A :meth:`get` (or a deduplicating :meth:`put`) marks an entry recently-used
  and **promotes** it back to the top tier -- hot prefixes stay fast.

Reuse index
-----------
A content key (e.g. ``hash_tokens(prefix)``) maps to a stored handle. ``has(key)``
is the prefix-reuse probe a serving layer runs before prefilling: a hit means the
KV is already resident somewhere in the hierarchy and can be loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from dexa.core.types import CompactCache, KVCache

# A tiered entry only needs to know how many bytes it occupies; both raw and
# compact caches answer that, so the store is agnostic to which it holds.
Cacheable = Union[KVCache, CompactCache]


@dataclass
class TierSpec:
    """One level of the memory hierarchy.

    ``capacity_bytes`` bounds the resident bytes; ``read_bandwidth_bytes_per_s``
    and ``fixed_latency_s`` define the modeled access cost
    ``fixed_latency_s + nbytes / read_bandwidth_bytes_per_s`` charged on every
    read/write touching this tier. Defaults span the usual three-orders-of-
    magnitude gap between HBM, DRAM-over-PCIe, and NVMe.
    """

    name: str
    capacity_bytes: int
    read_bandwidth_bytes_per_s: float
    fixed_latency_s: float = 0.0

    def access_seconds(self, nbytes: int) -> float:
        return self.fixed_latency_s + nbytes / max(1.0, self.read_bandwidth_bytes_per_s)


def default_tiers(
    gpu_bytes: int = 8 * 1024**2,
    cpu_bytes: int = 64 * 1024**2,
    nvme_bytes: int = 4 * 1024**3,
) -> list[TierSpec]:
    """A ``gpu`` -> ``cpu`` -> ``nvme`` hierarchy with realistic bandwidth gaps.

    Capacities default small enough that long runs spill and evict; override per
    call. Bandwidths approximate HBM (~2 TB/s), PCIe gen4 (~25 GB/s), and a
    consumer NVMe SSD (~3 GB/s)."""
    return [
        TierSpec("gpu", gpu_bytes, 2_000e9, 0.0),
        TierSpec("cpu", cpu_bytes, 25e9, 10e-6),
        TierSpec("nvme", nvme_bytes, 3e9, 100e-6),
    ]


@dataclass
class _Entry:
    handle: str
    key: str
    tenant: str
    cache: Cacheable
    nbytes: int
    tier: str
    last_used: int


class TieredCacheStore:
    """An LRU, tiered, content-addressed KV store (LMCache-style reuse + offload).

    Parameters
    ----------
    tiers:
        Ordered fast -> slow list of :class:`TierSpec`. Defaults to
        :func:`default_tiers`.
    """

    def __init__(self, tiers: Optional[list[TierSpec]] = None) -> None:
        self.tiers: list[TierSpec] = tiers or default_tiers()
        if not self.tiers:
            raise ValueError("TieredCacheStore needs at least one tier")
        self._tier_index = {t.name: i for i, t in enumerate(self.tiers)}

        self._entries: dict[str, _Entry] = {}        # handle -> entry
        self._index: dict[tuple[str, str], str] = {}  # (tenant, key) -> handle
        self._clock = 0

        self._used: dict[str, int] = {t.name: 0 for t in self.tiers}
        self._peak: dict[str, int] = {t.name: 0 for t in self.tiers}

        # counters
        self.reuse_lookups = 0
        self.reuse_hits = 0
        self.gets = 0
        self.get_hits = 0
        self.get_misses = 0
        self.demotions = 0
        self.promotions = 0
        self.evictions_dropped = 0
        self.modeled_access_seconds = 0.0

    # --- CacheStore protocol ---------------------------------------------
    def put(self, key: str, cache: Cacheable, *, tenant: str = "default") -> str:
        """Insert (or refresh) ``cache`` under content ``key`` for ``tenant``.

        Returns a stable handle. A second put of the same ``(tenant, key)`` is a
        deduplicating no-op on bytes -- it just marks the entry hot and promotes
        it -- which is the desired behavior when a request re-references a prefix
        that is already resident.
        """
        slot = (tenant, key)
        handle = self._index.get(slot)
        if handle is not None:
            entry = self._entries[handle]
            self._touch(entry)
            self._promote(entry)
            return handle

        handle = f"{tenant}::{key}"
        nbytes = int(cache.nbytes())
        top = self.tiers[0].name
        entry = _Entry(
            handle=handle, key=key, tenant=tenant, cache=cache,
            nbytes=nbytes, tier=top, last_used=self._tick(),
        )
        self._entries[handle] = entry
        self._index[slot] = handle
        self._used[top] += nbytes
        self._record_peak(top)
        self.modeled_access_seconds += self.tiers[0].access_seconds(nbytes)  # the write
        self._enforce_capacity()
        return handle

    def get(self, handle: str) -> Optional[Cacheable]:
        """Load the cache for ``handle``, charging the residing tier's access
        latency and promoting the entry back to the top tier (LRU + hot-data
        promotion). Returns ``None`` for an unknown/evicted handle."""
        self.gets += 1
        entry = self._entries.get(handle)
        if entry is None:
            self.get_misses += 1
            return None
        self.get_hits += 1
        self.modeled_access_seconds += self._tier(entry.tier).access_seconds(entry.nbytes)
        self._touch(entry)
        self._promote(entry)
        return entry.cache

    def has(self, key: str, *, tenant: str = "default") -> Optional[str]:
        """Prefix-reuse probe: return the handle if ``key`` is resident in *any*
        tier for ``tenant``, else ``None``. Records the lookup so :meth:`stats`
        can report a reuse hit rate."""
        self.reuse_lookups += 1
        handle = self._index.get((tenant, key))
        if handle is not None:
            self.reuse_hits += 1
        return handle

    def evict(self, handle: str) -> None:
        """Remove an entry entirely (no demotion). Safe on unknown handles."""
        entry = self._entries.pop(handle, None)
        if entry is None:
            return
        self._index.pop((entry.tenant, entry.key), None)
        self._used[entry.tier] -= entry.nbytes

    def stats(self) -> dict[str, Any]:
        """Snapshot of occupancy, peak retained bytes per tier, and the reuse /
        eviction / latency counters."""
        tiers = {
            t.name: {
                "capacity_bytes": int(t.capacity_bytes),
                "used_bytes": int(self._used[t.name]),
                "peak_bytes": int(self._peak[t.name]),
                "n_entries": sum(1 for e in self._entries.values() if e.tier == t.name),
            }
            for t in self.tiers
        }
        reuse_rate = self.reuse_hits / self.reuse_lookups if self.reuse_lookups else 0.0
        return {
            "tiers": tiers,
            "n_entries": len(self._entries),
            "total_bytes": int(sum(self._used.values())),
            "peak_total_bytes": int(sum(self._peak.values())),
            "peak_bytes_per_tier": {k: v["peak_bytes"] for k, v in tiers.items()},
            "reuse_lookups": self.reuse_lookups,
            "reuse_hits": self.reuse_hits,
            "reuse_hit_rate": float(reuse_rate),
            "gets": self.gets,
            "get_hits": self.get_hits,
            "get_misses": self.get_misses,
            "demotions": self.demotions,
            "promotions": self.promotions,
            "evictions_dropped": self.evictions_dropped,
            "modeled_access_seconds": float(self.modeled_access_seconds),
        }

    # --- internals --------------------------------------------------------
    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _touch(self, entry: _Entry) -> None:
        entry.last_used = self._tick()

    def _tier(self, name: str) -> TierSpec:
        return self.tiers[self._tier_index[name]]

    def _record_peak(self, tier: str) -> None:
        if self._used[tier] > self._peak[tier]:
            self._peak[tier] = self._used[tier]

    def _move(self, entry: _Entry, dst: str) -> None:
        self._used[entry.tier] -= entry.nbytes
        entry.tier = dst
        self._used[dst] += entry.nbytes
        self._record_peak(dst)

    def _promote(self, entry: _Entry) -> None:
        """Move ``entry`` to the top tier and re-enforce capacity downward."""
        top = self.tiers[0].name
        if entry.tier != top:
            self._move(entry, top)
            self.promotions += 1
            self._enforce_capacity()

    def _lru_in(self, tier: str) -> Optional[_Entry]:
        candidates = [e for e in self._entries.values() if e.tier == tier]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.last_used)

    def _enforce_capacity(self) -> None:
        """Cascade LRU eviction down the hierarchy; drop off the bottom tier."""
        last = len(self.tiers) - 1
        for i, tier in enumerate(self.tiers):
            while self._used[tier.name] > tier.capacity_bytes:
                victim = self._lru_in(tier.name)
                if victim is None:
                    break  # nothing here to move; capacity simply too small
                if i == last:
                    # off the bottom tier -> real eviction
                    self.evict(victim.handle)
                    self.evictions_dropped += 1
                else:
                    self._move(victim, self.tiers[i + 1].name)
                    self.demotions += 1
