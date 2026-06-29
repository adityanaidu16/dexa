"""The GTM cartridge benchmark, runnable on a real model.

Compiles a cartridge for a planted-fact corpus and compares three ways to put
that corpus "in context" -- full-context (prompt cache), TF-IDF RAG, and the
cartridge -- on quality, memory, per-query cost, and the break-even query count.
Prints the docs/CARTRIDGES.md-style table and saves the money plots.

Validated fast on CPU with the tiny-random Llama; the headline numbers come from
a real model on a GPU. Examples::

    # tiny smoke (CPU, seconds) -- plumbing only, numbers not meaningful:
    .venv/bin/python benchmarks/cartridge_bench.py \
        --model hf-internal-testing/tiny-random-LlamaForCausalLM \
        --facts 4 --filler 256 --t 16 --steps 4

    # the real artifact (run on GPU; SmolLM2 training is reserved off the dev box):
    .venv/bin/python benchmarks/cartridge_bench.py \
        --model HuggingFaceTB/SmolLM2-360M-Instruct \
        --facts 16 --filler 4000 --t 128 --steps 200 --rag-k 3

NOTE: do not run real SmolLM2 *training* in a dev loop on this machine -- the CPU
is reserved by another process. Use the tiny model for development.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexa.core.types import CostModel
from dexa.bench.corpus import make_corpus_qa, run_corpus_bench, report_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--facts", type=int, default=8, help="number of planted facts / queries")
    ap.add_argument("--filler", type=int, default=512, help="filler tokens around the facts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t", type=int, default=64, help="cartridge compact tokens")
    ap.add_argument("--steps", type=int, default=120, help="cartridge training steps")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-selfstudy", type=int, default=6)
    ap.add_argument("--answer-len", type=int, default=24)
    ap.add_argument("--rag-k", type=int, default=3)
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--n-decode", type=int, default=32, help="decoded tokens/query for the cost model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out-dir", default="benchmarks/out")
    args = ap.parse_args()

    from dexa.engine.hf_backend import HFBackend

    print(f"loading {args.model} ...", flush=True)
    backend = HFBackend(model_name=args.model, device=args.device, dtype=args.dtype)
    s = backend.spec
    print(f"  spec: {s.n_layers}L {s.n_q_heads}q/{s.n_kv_heads}kv head_dim={s.head_dim}", flush=True)

    corpus, qa = make_corpus_qa(
        backend, n_facts=args.facts, filler_tokens=args.filler, seed=args.seed
    )
    cartridge_opts = {
        "t": args.t, "steps": args.steps, "lr": args.lr,
        "n_selfstudy": args.n_selfstudy, "answer_len": args.answer_len,
        "verbose": True,
    }
    out_dir = Path(args.out_dir)
    results = run_corpus_bench(
        backend, corpus, qa,
        methods=["full_context", "rag", "cartridge"],
        cartridge_opts=cartridge_opts,
        rag_k=args.rag_k, chunk_tokens=args.chunk_tokens,
        cost=CostModel(), n_decode=args.n_decode,
        out_path=out_dir / "corpus.json", verbose=True,
    )
    print()
    report_corpus(results, out_dir=str(out_dir))


if __name__ == "__main__":
    main()
