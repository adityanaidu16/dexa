"""Tiering policy — where should an idle session's KV live?

The brain of the session service, and it's not heuristic hand-waving: the tier boundaries are
the break-even idle times derived from the measured cost model (evals/stateful_cost_model.py).
On each resume we know how long the session idled and how big its context is; this decides
whether to have kept it warm (GPU HBM), on RAM, on NVMe, or dropped (re-prefill).

Break-even for a tier = (re-prefill cost saved) / (that tier's $/hr to store the KV). Past a
tier's break-even, storing costs more than the re-prefill it saves, so demote or drop.
"""

from __future__ import annotations

from dataclasses import dataclass

# measured on A100-80GB, Qwen2.5-7B bf16 (evals/modal_stateful_session.py)
# context_tokens -> (kv_gb, prefill_ms, restore_cpu_ms, restore_nvme_ms)
PROFILE: dict[int, tuple[float, float, float, float]] = {
    4096:  (0.23, 296.0, 25.0, 246.0),
    16384: (0.94, 1321.0, 115.0, 390.0),
    32768: (1.88, 3167.0, 182.0, 646.0),
    65536: (3.76, 8567.0, 297.0, 1228.0),
}

# configurable rates (mirror stateful_cost_model.py)
GPU_USD_HR = 1.80
RAM_USD_GB_HR = 0.006
NVME_USD_GB_HR = 0.0002
GPU_USABLE_GB = 60.0
GPU_USD_S = GPU_USD_HR / 3600.0

TIERS = ("warm", "ram", "nvme", "drop")


def profile_for(tokens: int) -> tuple[float, float, float, float]:
    """Nearest-neighbour lookup with linear scaling for out-of-table sizes."""
    keys = sorted(PROFILE)
    if tokens <= keys[0]:
        base = PROFILE[keys[0]]; scale = tokens / keys[0]
        return (base[0] * scale, base[1] * scale, base[2] * scale, base[3] * scale)
    for lo, hi in zip(keys, keys[1:]):
        if tokens <= hi:
            f = (tokens - lo) / (hi - lo)
            a, b = PROFILE[lo], PROFILE[hi]
            return tuple(a[i] + f * (b[i] - a[i]) for i in range(4))  # type: ignore
    base = PROFILE[keys[-1]]; scale = tokens / keys[-1]  # extrapolate above the table
    return (base[0] * scale, base[1] * scale, base[2] * scale, base[3] * scale)


@dataclass
class Breakevens:
    warm_s: float
    ram_s: float
    nvme_s: float


def breakevens(tokens: int) -> Breakevens:
    kv_gb, prefill_ms, r_cpu_ms, r_nvme_ms = profile_for(tokens)
    prefill = prefill_ms / 1000 * GPU_USD_S
    r_cpu = r_cpu_ms / 1000 * GPU_USD_S
    r_nvme = r_nvme_ms / 1000 * GPU_USD_S
    warm_hr = kv_gb / GPU_USABLE_GB * GPU_USD_HR
    ram_hr = kv_gb * RAM_USD_GB_HR
    nvme_hr = kv_gb * NVME_USD_GB_HR
    return Breakevens(
        warm_s=prefill / warm_hr * 3600 if warm_hr else float("inf"),
        ram_s=(prefill - r_cpu) / ram_hr * 3600 if ram_hr else float("inf"),
        nvme_s=(prefill - r_nvme) / nvme_hr * 3600 if nvme_hr else float("inf"),
    )


@dataclass
class Decision:
    tier: str            # warm | ram | nvme | drop
    idle_s: float
    breakevens: Breakevens
    reason: str


class TieringPolicy:
    """Decide the storage tier for a session given how long it idled and its context size."""

    def decide(self, idle_s: float, tokens: int) -> Decision:
        be = breakevens(tokens)
        if idle_s <= be.warm_s:
            tier, why = "warm", f"idle {idle_s:.0f}s <= warm break-even {be.warm_s:.0f}s"
        elif idle_s <= be.ram_s:
            tier, why = "ram", f"idle {idle_s:.0f}s <= RAM break-even {be.ram_s:.0f}s"
        elif idle_s <= be.nvme_s:
            tier, why = "nvme", f"idle {idle_s:.0f}s <= NVMe break-even {be.nvme_s:.0f}s"
        else:
            tier, why = "drop", f"idle {idle_s:.0f}s past NVMe break-even {be.nvme_s:.0f}s — re-prefill"
        return Decision(tier=tier, idle_s=idle_s, breakevens=be, reason=why)


def estimated_savings(tokens: int, warm: bool = True) -> dict:
    """GPU compute a warm turn avoids: it restores the KV instead of re-prefilling the shared
    prefix. Modeled from the profile (prefill vs restore), independent of end-to-end HTTP
    latency (which also carries network + decode overhead in both cold and warm turns)."""
    _kv, prefill_ms, restore_ms, _rn = profile_for(tokens)
    if not warm:                          # a cold first-touch turn IS the prefill; nothing saved
        return {"reprefill_ms": round(prefill_ms, 1), "restore_ms": round(restore_ms, 1),
                "saved_ms": 0.0, "saved_usd": 0.0, "speedup": None}
    saved_ms = max(0.0, prefill_ms - restore_ms)
    return {
        "reprefill_ms": round(prefill_ms, 1),
        "restore_ms": round(restore_ms, 1),
        "saved_ms": round(saved_ms, 1),
        "saved_usd": round(saved_ms / 1000 * GPU_USD_S, 8),
        "speedup": round(prefill_ms / restore_ms, 1) if restore_ms > 0 else None,
    }
