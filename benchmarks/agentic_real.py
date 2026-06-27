"""Real-model long-horizon agentic demonstration for Dexa working memory.

The headline "long-context agentic" result. A simulated agent runs for many turns
on a *real* transformer, accumulating context with facts planted early. We then
ask about those early facts and compare memory strategies on late-recall *and*
memory footprint:

    full_kv          high recall, but KV memory grows unbounded (the memory wall)
    truncate_recent  bounded memory, but forgets early facts (low recall)
    dexa:<compactor> bounded memory AND high late-recall (the money result)

Recall is the needle-recall log-prob fraction (1.0 = full-KV ceiling, 0.0 =
no-context floor), the same metric as ``benchmarks/niah_real.py``.

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/agentic_real.py \
        --model HuggingFaceTB/SmolLM2-360M-Instruct \
        --turns 8 --turn-tokens 150 --budget 300 --seeds 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dexa.bench.agentic import DEFAULT_STRATEGIES, run_agentic


def _plot(summary: list[dict], out_path: Path) -> None:
    """Recall-vs-peak-memory scatter (guarded matplotlib import)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"(matplotlib unavailable, skipping plot: {e})")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s in summary:
        ax.scatter(s["peak_kv_mb"], s["late_recall"], s=90)
        ax.annotate(
            s["strategy"],
            (s["peak_kv_mb"], s["late_recall"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )
    ax.set_xlabel("peak KV memory (MB)  -- lower is better")
    ax.set_ylabel("late-recall fraction  -- higher is better")
    ax.set_title("Dexa working memory: long-horizon recall vs. memory")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--turn-tokens", type=int, default=150)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--keep-recent", type=int, default=None, help="default: budget // 2")
    ap.add_argument("--n-facts", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ref-bank-cap", type=int, default=64)
    ap.add_argument("--out-dir", default="benchmarks/out")
    args = ap.parse_args()

    from dexa.engine.hf_backend import HFBackend

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    backend = HFBackend(args.model)
    spec = backend.spec
    print(
        f"  spec: {spec.n_layers}L {spec.n_q_heads}q/{spec.n_kv_heads}kv "
        f"head_dim={spec.head_dim}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    result = run_agentic(
        backend,
        strategies=DEFAULT_STRATEGIES,
        n_turns=args.turns,
        turn_tokens=args.turn_tokens,
        budget_tokens=args.budget,
        keep_recent_tokens=args.keep_recent,
        n_facts=args.n_facts,
        seeds=args.seeds,
        ref_bank_cap=args.ref_bank_cap,
        out_path=str(out_dir / "agentic.json"),
        verbose=True,
    )

    summary = result["summary"]
    order = {s: i for i, s in enumerate(DEFAULT_STRATEGIES)}
    summary.sort(key=lambda s: order.get(s["strategy"], 99))

    print("\n=== agentic long-horizon memory results ===")
    print(f"{'strategy':>26s} | {'late-recall':>11s} | {'peak tok':>9s} | "
          f"{'peak KV MB':>10s} | {'compactions':>11s}")
    print("-" * 80)
    for s in summary:
        print(
            f"{s['strategy']:>26s} | {s['late_recall']:>11.2f} | {s['peak_tokens']:>9d} | "
            f"{s['peak_kv_mb']:>10.2f} | {s['n_compactions']:>11d}"
        )

    _plot(summary, out_dir / "agentic_tradeoff.png")
    print(f"\nraw -> {out_dir / 'agentic.json'}")


if __name__ == "__main__":
    main()
