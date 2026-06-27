"""Plot the needle-recall frontier from benchmarks/out/niah_real.json.

Produces benchmarks/out/niah_frontier.png: mean needle-recall fraction vs
compression ratio, one line per compaction method. The money plot — Attention
Matching should sit far above the selection baselines as compression grows.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["attention_matching", "heavy_hitter", "snapkv", "recent_window", "random_subset"]
LABELS = {
    "attention_matching": "Attention Matching (Dexa)",
    "heavy_hitter": "Heavy-Hitter (H2O)",
    "snapkv": "SnapKV",
    "recent_window": "Recent window",
    "random_subset": "Random subset",
}


def main(path: str = "benchmarks/out/niah_real.json", out: str = "benchmarks/out/niah_frontier.png") -> None:
    rows = json.loads(Path(path).read_text())
    # mean recall per (method, ratio)
    agg: dict = defaultdict(list)
    ratios: set = set()
    for r in rows:
        if r["method"] in ("full_kv", "no_context"):
            continue
        agg[(r["method"], r["ratio"])].append(r["recall_frac"])
        ratios.add(r["ratio"])
    ratios = sorted(ratios)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable ({e}); skipping plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ORDER:
        ys = [np.mean(agg[(m, rt)]) if agg[(m, rt)] else np.nan for rt in ratios]
        if all(np.isnan(ys)):
            continue
        ax.plot(ratios, ys, marker="o", label=LABELS.get(m, m), linewidth=2)
    ax.axhline(1.0, ls="--", c="green", alpha=0.6, label="Full-KV ceiling")
    ax.axhline(0.0, ls="--", c="grey", alpha=0.6, label="No-context floor")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{int(r)}x" for r in ratios])
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Needle-recall fraction")
    ax.set_title("Dexa: needle recall vs KV compression (SmolLM2-360M)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
