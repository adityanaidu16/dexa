"""Persist-and-resume benchmark — the system-layer wedge.

For a range of context lengths, compare resuming a session two ways:

* **cold** (what a stateless engine / ephemeral prefix cache does on a cache
  miss): re-prefill the whole context from scratch, then decode.
* **resume** (Dexa): load the persisted KV state from the store, then decode.

Reports, per length: time-to-first-token for each, the **speedup**, whether the
resumed output is **identical** to the live one (lossless), the state size, and
the idle-session memory story (persisted sessions cost ~0 GPU between turns).

The speedup grows with context length and model size, because cold pays O(n)
*compute* (prefill) while resume pays O(n) *I/O* (load) — and compute >> I/O at
length. On a small CPU model the absolute numbers are modest but the ratio and
the correctness guarantee are the point.
"""

from __future__ import annotations

import time
from typing import Optional

from dexa.core.types import CostModel
from dexa.session.store import SessionStore

_BASE = (
    "The quarterly review covered revenue, churn, and the migration plan in "
    "detail, with the team noting that the new pipeline reduced latency while "
    "the storage tier absorbed the additional load without incident. "
)


def _make_context(backend, target_tokens: int) -> list[int]:
    text = _BASE
    while len(backend.tokenize(text)) < target_tokens:
        text += _BASE
    return backend.tokenize(text)[:target_tokens]


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_persist_bench(
    backend,
    *,
    lengths=(256, 1024, 4096),
    gen_tokens: int = 8,
    store: Optional[SessionStore] = None,
    cost: Optional[CostModel] = None,
    compress: bool = False,
    keep_native: bool = False,
    verbose: bool = True,
) -> dict:
    store = store or SessionStore()
    cost = cost or CostModel()
    rows = []
    for L in lengths:
        ctx = _make_context(backend, L)
        T = len(ctx)

        # cold: re-prefill from scratch, then TTFT + a short continuation.
        kv_live, prefill_s = _time(lambda: backend.prefill(ctx))
        _, cold_ttft_s = _time(lambda: backend.generate(kv_live, [], max_new_tokens=1))
        live_cont = backend.generate(kv_live, [], max_new_tokens=gen_tokens)
        cold_resume_s = prefill_s + cold_ttft_s

        # persist, then drop the live cache (simulate teardown / preemption).
        sid = f"bench-{T}"
        meta = store.save(sid, kv_live, compress=compress)
        del kv_live

        # resume: load persisted state, then TTFT + the same continuation.
        # keep_native skips the bf16->fp32 host widen on load (the resume-latency
        # win); HFBackend reinterprets the bf16 bits straight to device.
        (kv_loaded, load_s) = store.load(sid, keep_native=keep_native)
        _, resume_ttft_s = _time(lambda: backend.generate(kv_loaded, [], max_new_tokens=1))
        resume_cont = backend.generate(kv_loaded, [], max_new_tokens=gen_tokens)
        resume_s = load_s + resume_ttft_s
        store.delete(sid)

        identical = resume_cont == live_cont
        speedup = cold_resume_s / resume_s if resume_s > 0 else float("inf")
        state_mb = meta["nbytes"] / 1e6
        rows.append({
            "length": T,
            "cold_resume_ms": cold_resume_s * 1e3,
            "resume_ms": resume_s * 1e3,
            "prefill_ms": prefill_s * 1e3,
            "load_ms": load_s * 1e3,
            "speedup": speedup,
            "identical_output": identical,
            "state_mb": state_mb,
            "state_kb_per_token": meta["nbytes"] / T / 1e3,
        })
        if verbose:
            ok = "OK" if identical else "MISMATCH!"
            print(f"  L={T:6d}  cold={cold_resume_s*1e3:8.1f}ms  resume={resume_s*1e3:8.1f}ms  "
                  f"speedup={speedup:5.1f}x  output={ok}  state={state_mb:.1f}MB", flush=True)

    summary = {
        "all_identical": all(r["identical_output"] for r in rows),
        "max_speedup": max((r["speedup"] for r in rows), default=0.0),
        "cost_model": cost.name,
    }
    return {"rows": rows, "summary": summary}


def run_compaction_persist_bench(
    backend,
    *,
    length: int = 2000,
    ratios=(8, 32, 128),
    compactor: str = "recent_window",
    store: Optional[SessionStore] = None,
    verbose: bool = True,
) -> dict:
    """Persist a long context RAW vs COMPACTED at several ratios; report state
    size + reload time. This is why compaction matters for portable state:
    moving an 8B 200k-token session means moving GBs of KV — compaction shrinks
    it ~ratio×, so reload (and cross-GPU/spot migration) gets ~ratio× cheaper.
    Quality is the separate compaction tradeoff (keep recent raw, compact cold).
    """
    import time as _t

    from dexa.compaction.base import CompactionBudget
    from dexa.compaction.baselines import build
    from dexa.session.state import load_compactcache, save_compactcache

    store = store or SessionStore()
    ctx = _make_context(backend, length)
    T = len(ctx)
    full = backend.prefill(ctx)

    m = store.save("raw", full)
    (_, raw_load) = store.load("raw")
    store.delete("raw")
    raw_mb = m["nbytes"] / 1e6
    rows = [{"ratio": 1, "t": T, "state_mb": raw_mb, "reload_ms": raw_load * 1e3,
             "size_reduction": 1.0, "reload_speedup": 1.0}]

    comp = build(compactor)
    for r in ratios:
        cc = comp.compact(full, CompactionBudget(ratio=float(r)))
        p = save_compactcache(cc, store.root / f"comp{r}")
        sz = p.stat().st_size
        t0 = _t.perf_counter()
        load_compactcache(p)
        ls = _t.perf_counter() - t0
        p.unlink()
        rows.append({
            "ratio": int(r), "t": cc.layers[0].keys[0].shape[0],
            "state_mb": sz / 1e6, "reload_ms": ls * 1e3,
            "size_reduction": raw_mb / (sz / 1e6),
            "reload_speedup": raw_load / ls if ls > 0 else float("inf"),
        })
        if verbose:
            print(f"  ratio={int(r):>4}x  state={sz/1e6:8.1f}MB  reload={ls*1e3:7.1f}ms  "
                  f"({raw_mb/(sz/1e6):.0f}x smaller, {raw_load/ls if ls>0 else 0:.0f}x faster reload)",
                  flush=True)
    return {"rows": rows, "length": T, "model": backend.spec.name,
            "raw_state_mb": raw_mb}


def report_persist(results: dict) -> str:
    rows = results["rows"]
    lines = ["", "Persist-and-resume (cold re-prefill vs Dexa state reload)", ""]
    lines.append(f"{'length':>8} {'cold ms':>10} {'resume ms':>10} {'speedup':>8} "
                 f"{'lossless':>9} {'state MB':>9}")
    for r in rows:
        lines.append(f"{r['length']:>8} {r['cold_resume_ms']:>10.1f} {r['resume_ms']:>10.1f} "
                     f"{r['speedup']:>7.1f}x {('yes' if r['identical_output'] else 'NO'):>9} "
                     f"{r['state_mb']:>9.1f}")
    s = results["summary"]
    lines += ["", f"all outputs identical: {s['all_identical']}   "
                  f"max speedup: {s['max_speedup']:.1f}x"]
    out = "\n".join(lines)
    print(out)
    return out
