"""In-memory usage + savings ledger (per-session and global).

MVP persistence: process memory. Swap for Redis/Postgres when self-serve accounts land;
the interface (record / snapshot) stays the same.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Agg:
    requests: int = 0
    cache_hits: int = 0            # exact-duplicate frames served from cache
    dexa_usd: float = 0.0
    baseline_usd: float = 0.0
    saved_usd: float = 0.0
    dexa_image_tokens: int = 0
    baseline_image_tokens: int = 0
    redundant_frac_sum: float = 0.0   # for averaging screen redundancy
    redundant_n: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def avg_redundancy(self) -> float:
        return self.redundant_frac_sum / self.redundant_n if self.redundant_n else 0.0

    @property
    def x_cheaper(self) -> float:
        return (self.baseline_usd / self.dexa_usd) if self.dexa_usd > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "dexa_usd": round(self.dexa_usd, 6),
            "baseline_usd": round(self.baseline_usd, 6),
            "saved_usd": round(self.saved_usd, 6),
            "saved_pct": round(100 * self.saved_usd / self.baseline_usd, 2) if self.baseline_usd else 0.0,
            "x_cheaper": round(self.x_cheaper, 2),
            "avg_screen_redundancy_pct": round(100 * self.avg_redundancy, 1),
            "dexa_image_tokens": self.dexa_image_tokens,
            "baseline_image_tokens": self.baseline_image_tokens,
        }


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = Agg()
        self.by_session: dict[str, Agg] = defaultdict(Agg)

    def record(self, session: str, *, saved, redundant_frac: float, cache_hit: bool) -> None:
        with self._lock:
            for agg in (self.total, self.by_session[session]):
                agg.requests += 1
                agg.cache_hits += int(cache_hit)
                agg.dexa_usd += saved.ours.usd
                agg.baseline_usd += saved.baseline.usd
                agg.saved_usd += saved.saved_usd
                agg.dexa_image_tokens += saved.ours.image_tokens
                agg.baseline_image_tokens += saved.baseline.image_tokens
                agg.redundant_frac_sum += redundant_frac
                agg.redundant_n += 1
                agg.last_seen = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total": self.total.as_dict(),
                "sessions": {s: a.as_dict() for s, a in self.by_session.items()},
                "session_count": len(self.by_session),
            }
