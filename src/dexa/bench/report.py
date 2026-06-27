"""Reporting: text tables + the money plots.

* :func:`aggregate_rows` -- collapse per-task rows into one row per
  (compactor, ratio), averaging numeric metrics.
* :func:`render_table` -- pretty table via ``rich`` if available, else plain
  text.
* :func:`plot_frontier` / :func:`plot_memory_saving` -- matplotlib figures
  saved under the output dir; guarded so reporting still works without
  matplotlib.
* :func:`render_report` -- do all of the above for a
  :class:`~dexa.bench.runner.MatrixResult`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

_NUMERIC_KEYS = (
    "recon_cosine",
    "recon_rel_l2",
    "acc_exact_match",
    "acc_token_f1",
    "compression_ratio",
    "memory_saving",
    "nbytes_saving",
    "compaction_seconds",
    "decode_gpu_seconds_compact",
    "decode_gpu_seconds_saved",
)


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and v == v]  # drop None/NaN
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate_rows(rows: Iterable[dict]) -> list[dict]:
    """Average numeric metrics across tasks, grouped by (compactor, ratio)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["compactor"], r["ratio"])].append(r)

    out: list[dict] = []
    for (compactor, ratio), grp in groups.items():
        agg = {"compactor": compactor, "ratio": ratio, "n": len(grp)}
        for k in _NUMERIC_KEYS:
            if any(k in r for r in grp):
                agg[k] = _mean([r.get(k) for r in grp])
        out.append(agg)
    # order: full_kv first, then by compactor then ratio
    out.sort(key=lambda r: (r["compactor"] != "full_kv", r["compactor"], r["ratio"]))
    return out


_COLUMNS = [
    ("compactor", "compactor", "{}"),
    ("ratio", "ratio", "{:.0f}x"),
    ("recon_rel_l2", "recon rel_l2", "{:.4f}"),
    ("recon_cosine", "recon cos", "{:.4f}"),
    ("acc_token_f1", "acc f1", "{:.3f}"),
    ("compression_ratio", "compress", "{:.1f}x"),
    ("memory_saving", "mem save", "{:.1%}"),
    ("compaction_seconds", "compact s", "{:.4f}"),
]


def _fmt(row: dict, key: str, fmt: str) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def render_table(agg: list[dict]) -> str:
    """Return a printable table (uses rich if installed, else plain text)."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Dexa compaction benchmark (mean over tasks)")
        for _, header, _ in _COLUMNS:
            table.add_column(header, justify="right", no_wrap=True)
        for row in agg:
            table.add_row(*[_fmt(row, k, f) for k, _, f in _COLUMNS])
        import io

        console = Console(record=True, width=140, file=io.StringIO())
        console.print(table)
        return console.export_text()
    except Exception:
        headers = [h for _, h, _ in _COLUMNS]
        widths = [len(h) for h in headers]
        cells = []
        for row in agg:
            c = [_fmt(row, k, f) for k, _, f in _COLUMNS]
            cells.append(c)
            widths = [max(w, len(x)) for w, x in zip(widths, c)]
        lines = ["  ".join(h.rjust(w) for h, w in zip(headers, widths))]
        lines.append("  ".join("-" * w for w in widths))
        for c in cells:
            lines.append("  ".join(x.rjust(w) for x, w in zip(c, widths)))
        return "\n".join(lines)


def plot_frontier(agg: list[dict], out_dir: str) -> str | None:
    """Quality vs compression frontier (one line per compactor). The money plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    by_comp: dict[str, list[dict]] = defaultdict(list)
    for r in agg:
        if r["compactor"] == "full_kv":
            continue
        by_comp[r["compactor"]].append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for comp, grp in sorted(by_comp.items()):
        grp = sorted(grp, key=lambda r: r["compression_ratio"])
        xs = [g["compression_ratio"] for g in grp]
        ys = [g["recon_rel_l2"] for g in grp]
        ax.plot(xs, ys, marker="o", label=comp)
    ax.set_xlabel("compression ratio (T / compact tokens)  -> more compression")
    ax.set_ylabel("attention recon error (rel L2)  -> lower is better")
    ax.set_title("Quality vs compression frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "frontier.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_memory_saving(agg: list[dict], out_dir: str) -> str | None:
    """Bar chart of memory saving per compactor × ratio."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    rows = [r for r in agg if r["compactor"] != "full_kv"]
    if not rows:
        return None
    comps = sorted({r["compactor"] for r in rows})
    ratios = sorted({r["ratio"] for r in rows})
    width = 0.8 / max(1, len(comps))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ratios))
    for i, comp in enumerate(comps):
        ys = []
        for ratio in ratios:
            match = [r for r in rows if r["compactor"] == comp and r["ratio"] == ratio]
            ys.append(match[0]["memory_saving"] if match else 0.0)
        ax.bar(x + i * width, ys, width, label=comp)
    ax.set_xticks(x + width * (len(comps) - 1) / 2)
    ax.set_xticklabels([f"{int(r)}x" for r in ratios])
    ax.set_xlabel("target compression ratio")
    ax.set_ylabel("KV memory saving")
    ax.set_title("Memory saving by compactor")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "memory_saving.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_report(result, out_dir: str = os.path.join("benchmarks", "out"), *, plots: bool = True) -> dict:
    """Aggregate, print a table, and (optionally) save plots. Returns a summary
    dict with the rendered table text and any plot paths."""
    rows = list(result)
    agg = aggregate_rows(rows)
    table = render_table(agg)
    print(table)

    paths: dict[str, str | None] = {}
    if plots:
        paths["frontier"] = plot_frontier(agg, out_dir)
        paths["memory_saving"] = plot_memory_saving(agg, out_dir)
        for name, p in paths.items():
            if p:
                print(f"saved {name} plot -> {p}")
        if not any(paths.values()):
            print("(matplotlib not available; skipped plots)")
    return {"table": table, "aggregate": agg, "plots": paths}
