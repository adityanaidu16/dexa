"""Persist-and-resume demo — the system-layer wedge.

Two modes:

  bench   : measure resume-latency vs cold re-prefill across context lengths,
            and verify the resumed output is identical (lossless).

  save / resume : the genuine "survives a restart" demo across SEPARATE
            processes. `save` prefills a context, persists the session, and
            records the continuation it would produce. `resume` (a fresh
            process — kill the box in between!) loads the state and continues,
            proving identical output with ~0 re-prefill.

Examples:
  python benchmarks/persist_demo.py bench  --model HuggingFaceTB/SmolLM2-360M-Instruct \
         --device auto --lengths 256,1024,4096
  python benchmarks/persist_demo.py save   --model ... --length 2000 --session demo
  # ... restart the pod / move GPUs ...
  python benchmarks/persist_demo.py resume --model ... --session demo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _backend(args):
    import torch
    from dexa.engine.hf_backend import HFBackend
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(f"loading {args.model} on {device}/{dtype} ...", flush=True)
    return HFBackend(model_name=args.model, device=device, dtype=dtype)


def _cmd_bench(args):
    from dexa.bench.persist import run_persist_bench, report_persist
    from dexa.session.store import SessionStore
    be = _backend(args)
    lengths = [int(x) for x in args.lengths.split(",")]
    store = SessionStore(args.store_dir)
    res = run_persist_bench(be, lengths=lengths, gen_tokens=args.gen, store=store)
    report_persist(res)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "persist.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nsaved -> {out/'persist.json'}")


def _cmd_save(args):
    from dexa.bench.persist import _make_context
    from dexa.session.store import SessionStore
    be = _backend(args)
    store = SessionStore(args.store_dir)
    ctx = _make_context(be, args.length)
    t0 = time.perf_counter()
    kv = be.prefill(ctx)
    prefill_s = time.perf_counter() - t0
    cont = be.generate(kv, [], max_new_tokens=args.gen)
    meta = store.save(args.session, kv)
    (Path(args.store_dir) / f"{args.session}.gold.json").write_text(
        json.dumps({"continuation": cont, "n_ctx": len(ctx)}))
    print(f"prefilled {len(ctx)} tokens in {prefill_s*1e3:.0f}ms, persisted "
          f"{meta['nbytes']/1e6:.1f}MB to session '{args.session}'.")
    print(f"continuation (for verification): {cont}")
    print("Now restart the box / move GPUs, then run: resume --session "
          f"{args.session}")


def _cmd_resume(args):
    from dexa.session.store import SessionStore
    be = _backend(args)
    store = SessionStore(args.store_dir)
    if not store.has(args.session):
        raise SystemExit(f"no persisted session '{args.session}' in {args.store_dir}")
    kv, load_s = store.load(args.session)
    t0 = time.perf_counter()
    cont = be.generate(kv, [], max_new_tokens=args.gen)
    gen_s = time.perf_counter() - t0
    gold = json.loads((Path(args.store_dir) / f"{args.session}.gold.json").read_text())
    identical = cont == gold["continuation"]
    print(f"resumed session '{args.session}' ({gold['n_ctx']} tokens) with ZERO "
          f"re-prefill: state load {load_s*1e3:.0f}ms + decode {gen_s*1e3:.0f}ms.")
    print(f"continuation: {cont}")
    print(f"identical to pre-restart output: {identical}  "
          f"{'(lossless resume confirmed)' if identical else '(MISMATCH)'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "save", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
        p.add_argument("--device", default="auto")
        p.add_argument("--store-dir", default=".dexa_sessions")
        p.add_argument("--gen", type=int, default=8)
        if name == "bench":
            p.add_argument("--lengths", default="256,1024,4096")
            p.add_argument("--out-dir", default="benchmarks/out")
        else:
            p.add_argument("--length", type=int, default=2000)
            p.add_argument("--session", default="demo")
    args = ap.parse_args()
    {"bench": _cmd_bench, "save": _cmd_save, "resume": _cmd_resume}[args.cmd](args)


if __name__ == "__main__":
    main()
