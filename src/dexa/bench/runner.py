"""Benchmark matrix runner.

``run_matrix`` prefills each task once, derives reference queries once (split
into a compaction set the compactor may see and a held-out eval set used only
for scoring), then sweeps every compactor × ratio cell. It always emits a
``FullKV`` reference row per task (recon error 0, compression 1) so tables and
the frontier plot have an anchor. Results are plain dicts saved to
``benchmarks/out/results.json`` -- no pandas required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from dexa.bench import _compactors
from dexa.bench.metrics import _Timer, answer_accuracy, attention_recon_error, system_metrics
from dexa.bench.tasks import Task
from dexa.compaction.base import CompactionBudget
from dexa.core.types import CostModel, RefQueries

DEFAULT_OUT_DIR = os.path.join("benchmarks", "out")
DEFAULT_COMPACTORS = ["full_kv", "random_subset", "recent_window", "attention_matching"]
DEFAULT_RATIOS = [2.0, 4.0, 8.0, 16.0]


@dataclass
class MatrixResult:
    """Rows plus the cost model used; behaves like a thin DataFrame stand-in."""

    rows: list[dict]
    cost: CostModel

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def filter(self, **kw) -> list[dict]:
        return [r for r in self.rows if all(r.get(k) == v for k, v in kw.items())]

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"cost_model": self.cost.__dict__, "rows": self.rows}, f, indent=2
            )
        return path


def _split_refs(refs: RefQueries, eval_frac: float = 0.3) -> tuple[RefQueries, RefQueries]:
    """Split reference queries along the query axis into (compaction, eval)."""
    layers = refs.layers
    if not layers:
        return refs, refs
    n = layers[0].shape[1]
    n_eval = max(1, int(round(n * eval_frac)))
    n_eval = min(n_eval, n - 1) if n > 1 else n
    comp_layers = [L[:, : n - n_eval] for L in layers]
    eval_layers = [L[:, n - n_eval :] for L in layers]
    comp = RefQueries(spec=refs.spec, layers=comp_layers)
    ev = RefQueries(spec=refs.spec, layers=eval_layers)
    return comp, ev


def _row_base(task: Task) -> dict:
    return {
        "task": task.name,
        "length": task.meta.get("length"),
        "seed": task.meta.get("seed"),
        "context_tokens": len(task.context_ids),
    }


def run_matrix(
    backend,
    compactors: list[str] = DEFAULT_COMPACTORS,
    ratios: list[float] = DEFAULT_RATIOS,
    tasks: Optional[list[Task]] = None,
    *,
    ref_strategy: str = "repeat_prefill",
    cost: Optional[CostModel] = None,
    out_path: Optional[str] = os.path.join(DEFAULT_OUT_DIR, "results.json"),
    score_accuracy: bool = True,
) -> MatrixResult:
    """Run the full compactor × ratio × task matrix.

    Returns a :class:`MatrixResult`. If ``out_path`` is set, raw rows are saved
    there as JSON.
    """
    if tasks is None:
        from dexa.bench.tasks import make_tasks

        tasks = make_tasks(backend, lengths=[256], n_per=1)
    cost = cost or CostModel()

    rows: list[dict] = []
    for task in tasks:
        full = backend.prefill(task.context_ids)
        refs = backend.reference_queries(task.context_ids, strategy=ref_strategy)
        comp_refs, eval_refs = _split_refs(refs)

        # --- FullKV reference row (always present) ---
        sysm = system_metrics(cost, full, None, 0.0)
        row = _row_base(task)
        row.update(
            {
                "compactor": "full_kv",
                "ratio": 1.0,
                "recon_cosine": 1.0,
                "recon_rel_l2": 0.0,
            }
        )
        if score_accuracy:
            row.update({f"acc_{k}": v for k, v in answer_accuracy(backend, full, task).items()})
        row.update(sysm)
        rows.append(row)

        # --- compactor × ratio sweep ---
        for name in compactors:
            if name == "full_kv":
                continue  # handled above as the reference row
            comp = _compactors.build(name)
            use_refs = comp_refs if getattr(comp, "needs_ref_queries", False) else None
            for ratio in ratios:
                budget = CompactionBudget(ratio=float(ratio))
                with _Timer() as t:
                    cc = comp.compact(full, budget, ref_queries=use_refs)
                quality = attention_recon_error(backend, full, cc, eval_refs)
                sysm = system_metrics(cost, full, cc, t.seconds)
                row = _row_base(task)
                row.update(
                    {
                        "compactor": name,
                        "ratio": float(ratio),
                        "recon_cosine": quality["cosine"],
                        "recon_rel_l2": quality["rel_l2"],
                    }
                )
                if score_accuracy:
                    row.update(
                        {f"acc_{k}": v for k, v in answer_accuracy(backend, cc, task).items()}
                    )
                row.update(sysm)
                rows.append(row)

    result = MatrixResult(rows=rows, cost=cost)
    if out_path:
        result.save(out_path)
    return result
