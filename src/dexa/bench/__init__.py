"""Dexa benchmark harness.

Backend-agnostic tooling that proves the compaction thesis: a small compact
cache preserves model behavior. The same code runs against the torch-free
:class:`~dexa.engine.fake.FakeBackend` (CI / plumbing + attention-reconstruction
quality) and a real HF backend (real answer accuracy).

Public surface::

    from dexa.bench import make_tasks, run_matrix, render_report
"""

from dexa.bench.tasks import (
    Task,
    make_tasks,
    niah_single,
    niah_multikey,
    multihop,
    synthetic_qa,
)
from dexa.bench.metrics import (
    attention_recon_error,
    answer_accuracy,
    system_metrics,
)
from dexa.bench.runner import run_matrix
from dexa.bench.report import render_report, aggregate_rows

__all__ = [
    "Task",
    "make_tasks",
    "niah_single",
    "niah_multikey",
    "multihop",
    "synthetic_qa",
    "attention_recon_error",
    "answer_accuracy",
    "system_metrics",
    "run_matrix",
    "render_report",
    "aggregate_rows",
]
