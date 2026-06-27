"""Dexa command-line entry point (``dexa`` script).

Subcommands
    bench   Run a compactor x ratio benchmark matrix and print the report.
            Defaults to the torch-free FakeBackend (CI / plumbing +
            attention-reconstruction quality). Pass ``--backend hf --model ...``
            to use a real HF backend if it is installed.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_backend(args):
    if args.backend == "fake":
        from dexa.engine.fake import FakeBackend

        return FakeBackend()
    if args.backend == "hf":
        try:
            from dexa.engine.hf_backend import HFBackend  # type: ignore
        except Exception as e:  # pragma: no cover - hf backend optional
            print(
                f"error: HF backend unavailable ({e}). Install extras with "
                "`pip install -e .[torch]` and ensure torch/transformers are present.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not args.model:
            print("error: --model is required with --backend hf", file=sys.stderr)
            raise SystemExit(2)
        return HFBackend(args.model)
    raise SystemExit(f"unknown backend {args.backend!r}")


def _cmd_bench(args) -> int:
    from dexa.bench.runner import DEFAULT_COMPACTORS, run_matrix
    from dexa.bench.tasks import make_tasks
    from dexa.bench.report import render_report

    backend = _build_backend(args)

    lengths = args.lengths or ([256, 1024] if args.backend == "fake" else [512])
    ratios = args.ratios or [2.0, 4.0, 8.0, 16.0]
    compactors = args.compactors or DEFAULT_COMPACTORS

    tasks = make_tasks(backend, lengths=lengths, n_per=args.n_per)
    out_dir = args.out_dir
    out_path = os.path.join(out_dir, "results.json")

    print(
        f"running matrix: backend={args.backend} tasks={len(tasks)} "
        f"compactors={compactors} ratios={ratios}"
    )
    result = run_matrix(
        backend,
        compactors=compactors,
        ratios=ratios,
        tasks=tasks,
        out_path=out_path,
        score_accuracy=not args.no_accuracy,
    )
    print(f"saved raw results -> {out_path}\n")
    render_report(result, out_dir=out_dir, plots=not args.no_plots)
    return 0


def _cmd_run(args) -> int:
    from dexa.bench.run import run_config_file

    run_config_file(args.config)
    return 0


def _float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dexa", description="Dexa inference-state engine CLI")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the compaction benchmark matrix")
    b.add_argument("--backend", choices=["fake", "hf"], default="fake")
    b.add_argument("--model", default=None, help="HF model id (with --backend hf)")
    b.add_argument("--lengths", type=_int_list, default=None, help="comma list of context lengths")
    b.add_argument("--ratios", type=_float_list, default=None, help="comma list of compression ratios")
    b.add_argument("--compactors", type=_str_list, default=None, help="comma list of compactor names")
    b.add_argument("--n-per", type=int, default=2, dest="n_per", help="tasks per (generator,length)")
    b.add_argument("--out-dir", default=os.path.join("benchmarks", "out"), dest="out_dir")
    b.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    b.add_argument("--no-accuracy", action="store_true", help="skip greedy-generate accuracy")
    b.set_defaults(func=_cmd_bench)

    r = sub.add_parser("run", help="run a config-driven benchmark (the cluster entrypoint)")
    r.add_argument("--config", required=True, help="path to a YAML/JSON RunConfig")
    r.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
