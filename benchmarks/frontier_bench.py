"""Run the accuracy-vs-KV-memory frontier benchmark (docs/BENCHMARK.md).

The decisive experiment: does a cartridge dominate the accuracy/memory frontier
vs full-context, RAG, and training-free KV compression? Needs a real model + GPU
for a real result; runs on CPU/tiny-model for plumbing.

Examples:
  python benchmarks/frontier_bench.py --model unsloth/Llama-3.1-8B-Instruct \
      --dataset ruler --task niah_single --length 4000 --n 20 --device auto
  python benchmarks/frontier_bench.py --model ... --dataset longbench \
      --subset multifieldqa_en --n 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _backend(model, device):
    import torch
    from dexa.engine.hf_backend import HFBackend
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(f"loading {model} on {device}/{dtype} ...", flush=True)
    return HFBackend(model_name=model, device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dataset", choices=["ruler", "longbench"], default="ruler")
    ap.add_argument("--task", default="niah_single", help="ruler task")
    ap.add_argument("--subset", default="multifieldqa_en", help="longbench subset")
    ap.add_argument("--length", type=int, default=4000, help="ruler context length")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ratios", default="4,16,50,128")
    ap.add_argument("--rag-ks", default="1,3,8")
    ap.add_argument("--cart-steps", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--out-dir", default="benchmarks/out")
    args = ap.parse_args()

    from dexa.bench.datasets import load_ruler, load_longbench
    from dexa.bench.frontier import run_frontier, report_frontier

    be = _backend(args.model, args.device)
    if args.dataset == "ruler":
        examples = load_ruler(task=args.task, length=args.length, n=args.n)
    else:
        examples = load_longbench(subset=args.subset, n=args.n)
    print(f"loaded {len(examples)} {args.dataset} examples", flush=True)

    res = run_frontier(
        be, examples,
        ratios=[float(x) for x in args.ratios.split(",")],
        rag_ks=[int(x) for x in args.rag_ks.split(",")],
        cartridge_opts={"steps": args.cart_steps},
        max_new_tokens=args.max_new,
    )
    report_frontier(res, out_dir=args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "frontier.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"saved {args.out_dir}/frontier.json")


if __name__ == "__main__":
    main()
