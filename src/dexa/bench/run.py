"""Config-driven run orchestrator: load a RunConfig, run its suites, write
results + a markdown summary + plots. This is what `dexa run --config ...` calls
and what the cluster scripts invoke.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dexa.bench.config import RunConfig, build_backend, load_config
from dexa.bench.suites import run_niah_recall
from dexa.core.types import CostModel


def run_config(cfg: RunConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"== Dexa run '{cfg.name}': backend={cfg.backend} model={cfg.model} "
          f"device={cfg.device} dtype={cfg.dtype} ==", flush=True)
    t0 = time.time()
    backend = build_backend(cfg)
    spec = backend.spec
    print(f"   loaded: {spec.n_layers}L {spec.n_q_heads}q/{spec.n_kv_heads}kv "
          f"head_dim={spec.head_dim} ({time.time()-t0:.1f}s)", flush=True)
    cost = CostModel(**cfg.cost) if cfg.cost else CostModel()

    results: dict = {"config": cfg.name, "model": cfg.model, "backend": cfg.backend}

    if cfg.niah.enabled:
        print("\n-- suite: niah recall --", flush=True)
        s = cfg.niah
        niah = run_niah_recall(
            backend, lengths=s.lengths, ratios=s.ratios, seeds=s.seeds, task=s.task,
            ref_strategy=s.ref_strategy, n_ref=s.n_ref, compactors=s.compactors,
            am_opts=s.am,
        )
        results["niah"] = niah
        (out / "niah.json").write_text(json.dumps(niah, indent=2))

    if cfg.agentic.enabled:
        print("\n-- suite: agentic --", flush=True)
        from dexa.bench.agentic import run_agentic
        a = cfg.agentic
        agentic = run_agentic(
            backend, strategies=a.strategies, n_turns=a.turns, turn_tokens=a.turn_tokens,
            budget_tokens=a.budget, n_facts=a.n_facts, seeds=a.seeds,
            ref_strategy=a.ref_strategy, cost=cost,
            out_path=str(out / "agentic.json"), verbose=True,
        )
        results["agentic"] = agentic

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    _write_report(cfg, results, out)
    if cfg.plots:
        _plots(cfg, results, out)
    print(f"\n== done in {time.time()-t0:.1f}s -> {out} ==", flush=True)
    return results


def _write_report(cfg: RunConfig, results: dict, out: Path) -> None:
    lines = [f"# Dexa run: {cfg.name}", "", f"- model: `{cfg.model}` (backend `{cfg.backend}`)", ""]
    if "niah" in results:
        lines += ["## Needle recall (1.0 = full-KV, 0.0 = no context)", ""]
        summ = results["niah"]["summary"]
        methods = [k.split(":", 1)[1] for k in summ[0] if k.startswith("recall:")] if summ else []
        lines.append("| ratio | " + " | ".join(methods) + " | AM−HH | AM wins |")
        lines.append("|" + "---|" * (len(methods) + 3))
        for r in summ:
            cells = [f"{r.get('recall:'+m, float('nan')):.2f}" for m in methods]
            d = r.get("am_minus_hh_mean")
            w = r.get("am_wins_frac")
            lines.append(f"| {int(r['ratio'])}x | " + " | ".join(cells)
                         + f" | {d:+.3f} | {w:.0%} |" if d is not None
                         else f"| {int(r['ratio'])}x | " + " | ".join(cells) + " | – | – |")
        lines.append("")
    if "agentic" in results:
        lines += ["## Long-horizon agentic (bounded memory vs recall)", ""]
        lines.append("| strategy | late-recall | peak tokens | peak KV MB | compactions |")
        lines.append("|---|---|---|---|---|")
        for r in results["agentic"]["summary"]:
            lines.append(f"| {r.get('strategy')} | {r.get('late_recall', r.get('recall', 0)):.2f} "
                         f"| {r.get('peak_tokens','-')} | {r.get('peak_kv_mb', r.get('peak_kv_bytes',0)/1e6):.1f} "
                         f"| {r.get('n_compactions','-')} |")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))


def _plots(cfg: RunConfig, results: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"   (plots skipped: {e})")
        return
    if "niah" in results:
        summ = results["niah"]["summary"]
        methods = [k.split(":", 1)[1] for k in summ[0] if k.startswith("recall:")] if summ else []
        ratios = [r["ratio"] for r in summ]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in methods:
            ax.plot(ratios, [r.get("recall:" + m, np.nan) for r in summ], marker="o", label=m)
        ax.axhline(1.0, ls="--", c="green", alpha=0.5)
        ax.axhline(0.0, ls="--", c="grey", alpha=0.5)
        ax.set_xscale("log", base=2); ax.set_xticks(ratios)
        ax.set_xticklabels([f"{int(r)}x" for r in ratios])
        ax.set_xlabel("compression ratio"); ax.set_ylabel("needle-recall fraction")
        ax.set_title(f"Dexa needle recall — {cfg.model}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out / "niah_frontier.png", dpi=130)
        print(f"   saved {out/'niah_frontier.png'}")


def run_config_file(path: str) -> dict:
    return run_config(load_config(path))
