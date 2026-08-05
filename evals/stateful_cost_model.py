"""Residency cost model — is a warm session cheaper than re-prefill, and when?

Turns the MEASURED restore/prefill numbers (modal_stateful_session.py, confirmed live by
modal_lmcache_restore.py) into economics. The question: on a resume, a stateless provider
re-prefills (burns GPU compute); a stateful provider pays to store the KV during the idle gap
plus a cheap restore. Which is cheaper, on which storage tier, for how long an idle gap?

Two values are modelled separately:
  * $  cost     — GPU-seconds saved by not re-prefilling, vs the $/GB-hr of holding the KV.
  * latency     — resume time (prefill vs restore); a pure product win, tier-independent.

Everything is transparent and configurable; run it and change the rates at the top.

    python evals/stateful_cost_model.py
"""

from __future__ import annotations

# ---- configurable rates (edit these) -----------------------------------------------------
GPU_USD_HR   = 1.80     # A100-80GB on-demand-ish; Modal lists higher, spot lower
RAM_USD_GB_HR  = 0.006  # cloud RAM (bundled w/ vCPU on mem-optimized instances)
NVME_USD_GB_HR = 0.0002 # local NVMe (~$0.10/GB-month)
GPU_USABLE_GB  = 60.0   # HBM available for KV after weights, for the warm-tier opportunity cost
FP8_KV = False          # halve KV bytes (fp8 KV cache) -> halves storage cost

# ---- measured data (modal_stateful_session.py, A100-80GB, Qwen2.5-7B bf16) ----------------
# kv_gb, prefill_ms, restore_cpu_ms, restore_nvme_ms
DATA = {
    4096:  dict(kv_gb=0.23, prefill_ms=296,  restore_cpu_ms=25,  restore_nvme_ms=246),
    16384: dict(kv_gb=0.94, prefill_ms=1321, restore_cpu_ms=115, restore_nvme_ms=390),
    32768: dict(kv_gb=1.88, prefill_ms=3167, restore_cpu_ms=182, restore_nvme_ms=646),
    65536: dict(kv_gb=3.76, prefill_ms=8567, restore_cpu_ms=297, restore_nvme_ms=1228),
}

GPU_USD_S = GPU_USD_HR / 3600.0


def fmt_time(hours: float) -> str:
    if hours == float("inf"):
        return "never pays"
    m = hours * 60
    if m < 90:
        return f"{m:.0f} min"
    if hours < 48:
        return f"{hours:.1f} hr"
    return f"{hours/24:.1f} days"


def analyze():
    print("\n" + "=" * 92)
    print(f"STATEFUL RESIDENCY COST MODEL   (GPU ${GPU_USD_HR}/hr, RAM ${RAM_USD_GB_HR}/GB-hr, "
          f"NVMe ${NVME_USD_GB_HR}/GB-hr{', fp8 KV' if FP8_KV else ''})")
    print("=" * 92)

    # ---- 1. per-resume cost + latency --------------------------------------------------
    print("\n1) PER-RESUME: re-prefill (stateless) vs restore (stateful), $ and latency")
    print(f"   {'context':>8} {'prefill $':>11} {'restore$ CPU':>13} {'restore$ NVMe':>14}"
          f" {'latency win':>13}")
    for C, d in DATA.items():
        pf = d["prefill_ms"] / 1000 * GPU_USD_S
        rc = d["restore_cpu_ms"] / 1000 * GPU_USD_S
        rn = d["restore_nvme_ms"] / 1000 * GPU_USD_S
        lat = d["prefill_ms"] / d["restore_cpu_ms"]
        print(f"   {C:>8} {pf*1e6:>9.1f}µ$ {rc*1e6:>11.1f}µ$ {rn*1e6:>12.1f}µ$ "
              f"{lat:>10.1f}×")

    # ---- 2. break-even idle time per tier ----------------------------------------------
    print("\n2) BREAK-EVEN IDLE: how long a session can idle before storing the KV costs more")
    print("   than the re-prefill it saves. Beyond this, drop the KV and re-prefill instead.")
    print(f"   {'context':>8} {'warm (GPU HBM)':>16} {'RAM':>12} {'NVMe':>14}")
    tiers = []
    for C, d in DATA.items():
        kv = d["kv_gb"] * (0.5 if FP8_KV else 1.0)
        pf = d["prefill_ms"] / 1000 * GPU_USD_S
        rc = d["restore_cpu_ms"] / 1000 * GPU_USD_S
        rn = d["restore_nvme_ms"] / 1000 * GPU_USD_S
        warm_hr = kv / GPU_USABLE_GB * GPU_USD_HR       # opportunity cost of HBM
        ram_hr  = kv * RAM_USD_GB_HR
        nvme_hr = kv * NVME_USD_GB_HR
        be_warm = pf / warm_hr if warm_hr else float("inf")            # warm restore ~0
        be_ram  = (pf - rc) / ram_hr if ram_hr else float("inf")
        be_nvme = (pf - rn) / nvme_hr if nvme_hr else float("inf")
        tiers.append((C, be_warm, be_ram, be_nvme))
        print(f"   {C:>8} {fmt_time(be_warm):>16} {fmt_time(be_ram):>12} {fmt_time(be_nvme):>14}")

    # ---- 3. tiering policy (which tier is cheapest at a given idle gap) -----------------
    print("\n3) TIERING POLICY at 64k context (cheapest option vs idle-gap length)")
    d = DATA[65536]; kv = d["kv_gb"] * (0.5 if FP8_KV else 1.0)
    pf = d["prefill_ms"] / 1000 * GPU_USD_S
    rc = d["restore_cpu_ms"] / 1000 * GPU_USD_S
    rn = d["restore_nvme_ms"] / 1000 * GPU_USD_S
    warm_hr, ram_hr, nvme_hr = kv/GPU_USABLE_GB*GPU_USD_HR, kv*RAM_USD_GB_HR, kv*NVME_USD_GB_HR
    print(f"   {'idle gap':>10} {'stateless':>11} {'warm':>9} {'RAM':>9} {'NVMe':>9}   {'winner':>10}")
    for g_min in (1, 5, 15, 60, 240, 1440):
        g = g_min / 60
        opts = {"stateless": pf, "warm": rc*0 + warm_hr*g, "RAM": rc + ram_hr*g,
                "NVMe": rn + nvme_hr*g}
        win = min(opts, key=opts.get)
        print(f"   {g_min:>7} min {opts['stateless']*1e6:>9.1f}µ {opts['warm']*1e6:>7.1f}µ "
              f"{opts['RAM']*1e6:>7.1f}µ {opts['NVMe']*1e6:>7.1f}µ   {win:>10}")

    # ---- 4. session projection ---------------------------------------------------------
    print("\n4) SESSION PROJECTION: 64k context, 50 turns, cost of the whole run")
    print(f"   {'idle/turn':>10} {'stateless $':>12} {'best stateful $':>16} {'saving':>8} {'tier':>7}")
    for g_min in (2, 15, 120):
        g = g_min / 60; T = 50
        stateless = T * pf
        cand = {"warm": pf + (T-1)*(warm_hr*g),
                "RAM":  pf + (T-1)*(rc + ram_hr*g),
                "NVMe": pf + (T-1)*(rn + nvme_hr*g)}
        tier = min(cand, key=cand.get); best = cand[tier]
        print(f"   {g_min:>7} min {stateless*1e3:>10.2f}m$ {best*1e3:>14.2f}m$ "
              f"{stateless/best:>6.1f}× {tier:>7}")

    print("\n" + "=" * 92)
    print("READ: latency win is large and tier-independent (always worth it for interactive")
    print("agents). The $ win needs the right tier — warm GPU pays only for ~minutes of idle,")
    print("RAM for minutes-to-tens-of-minutes, NVMe for hours. Policy: keep warm while active,")
    print("demote to RAM then NVMe as idle grows, drop past the NVMe break-even. fp8 KV (toggle")
    print("at top) ~doubles every break-even by halving stored bytes.")
    print("=" * 92)


if __name__ == "__main__":
    analyze()
