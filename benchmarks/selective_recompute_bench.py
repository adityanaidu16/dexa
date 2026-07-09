"""Phase 1 / Layer C benchmark: does selective KV recompute (CacheBlend) recover
downstream quality by recomputing only a small fraction of tokens — and does
High-KV-Deviation (HKVD) selection beat recency / random?

Setup. A length-preserving mid-context edit: a middle segment is replaced with
different content of the *same token length*, so the trailing suffix keeps its
positions but its cross-attention to the edited region is now stale. We reuse the
stale suffix KV and recompute the mandatory edited region plus a fraction of suffix
tokens chosen by each strategy, then measure how much of the downstream error
(vs a full re-prefill) is removed.

Metric. Mean L2 error of the **attention output** (queries attending over the whole
cache — a behavioral, softmax+value signal, not just KV norm) between the blended
cache and a full re-prefill, normalized so full-reuse (recompute 0%) = 1.0 and
full-recompute = 0.0. Lower is better. The claim under test: HKVD's curve drops
fastest — recomputing ~10-15% recovers most of the quality.

  python benchmarks/selective_recompute_bench.py --model hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from __future__ import annotations

import argparse

import numpy as np

from dexa.segment import Segment, SegmentedContext


def _backend(model, device):
    import torch
    from dexa.engine.hf_backend import HFBackend
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(f"loading {model} on {device}/{dtype} ...", flush=True)
    return HFBackend(model_name=model, device=device, dtype=dtype)


def _seg(be, name, text, role="context"):
    return Segment(name=name, token_ids=tuple(be.tokenize(text)), role=role)


def _equal_len(be, text_a, text_b):
    """Tokenize both, trim to the shorter length so an A->B swap is length-preserving."""
    a, b = be.tokenize(text_a), be.tokenize(text_b)
    n = min(len(a), len(b))
    return tuple(a[:n]), tuple(b[:n])


_L = ("The quick brown fox jumps over the lazy dog near the river bank at dawn. ")
_M = ("A distant galaxy spins in silence while comets trace long arcs of frozen light. ")


def _attn_error(be, blended, correct, queries):
    ao_b = be.attention_outputs(blended, queries)
    ao_c = be.attention_outputs(correct, queries)
    return float(np.mean([np.linalg.norm(b - c) for b, c in zip(ao_b, ao_c)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reps", type=int, default=3, help="doc repeats -> longer suffix")
    ap.add_argument("--gen", type=int, default=16, help="greedy tokens for the behavioral metric")
    args = ap.parse_args()

    be = _backend(args.model, args.device)

    doc_a, doc_b = _equal_len(be, "Edited region A: " + _L * 2, "Edited region B: " + _M * 2)
    prefix = _seg(be, "system", "You are an assistant with a large working context.", "system")
    edited_prev = Segment(name="mid", token_ids=doc_a, role="doc")
    edited_new = Segment(name="mid", token_ids=doc_b, role="doc")
    suffix = _seg(be, "suffix", ("Context body: " + _L * args.reps), "doc")

    prev_ctx = SegmentedContext([prefix, edited_prev, suffix])
    new_ctx = SegmentedContext([prefix, edited_new, suffix])
    assert prev_ctx.n_tokens == new_ctx.n_tokens

    prev_kv = be.prefill(prev_ctx.token_ids)
    correct = be.prefill(new_ctx.token_ids)
    queries = be.reference_queries(new_ctx.token_ids, strategy="self")

    # baselines: full reuse (recompute only mandatory edited region, 0% of suffix)
    # and full recompute.
    reuse0, s0 = be.recompute_selective(prev_kv, prev_ctx, new_ctx, recompute_frac=0.0)
    base_err = _attn_error(be, reuse0, correct, queries)
    print(f"\ncontext {new_ctx.n_tokens} tokens | edited region {len(doc_a)} | "
          f"stale suffix {suffix.n_tokens} | mandatory recompute {s0['mandatory_recompute']}")
    print(f"full-reuse attn error (baseline) = {base_err:.4e}  -> normalized to 1.000\n")

    # behavioral target: the greedy continuation a full re-prefill would produce.
    gold = be.generate(correct, [], max_new_tokens=args.gen)

    fracs = [0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0]
    strategies = ["hkvd", "recent", "random"]
    err_rows = {s: [] for s in strategies}
    agree_rows = {s: [] for s in strategies}
    pct_col = []
    for frac in fracs:
        for strat in strategies:
            blended, st = be.recompute_selective(
                prev_kv, prev_ctx, new_ctx, recompute_frac=frac, select=strat)
            err = _attn_error(be, blended, correct, queries)
            err_rows[strat].append(err / base_err if base_err else 0.0)
            gen = be.generate(blended, [], max_new_tokens=args.gen)
            match = sum(1 for a, b in zip(gen, gold) if a == b) / max(len(gold), 1)
            agree_rows[strat].append(match)
        pct_col.append(100 * st["recompute_fraction"])   # total %, strategy-independent

    print("[1] downstream attention-output error remaining (lower=better; KV-reconstruction, harsh)")
    print(f"{'recompute %':>12}", *[f"{s:>10}" for s in strategies])
    for i, pct in enumerate(pct_col):
        print(f"{pct:>11.0f}%", *[f"{err_rows[s][i]:>10.3f}" for s in strategies])

    print("\n[2] greedy continuation matches full re-prefill (higher=better; BEHAVIORAL — the real quality)")
    print(f"{'recompute %':>12}", *[f"{s:>10}" for s in strategies])
    for i, pct in enumerate(pct_col):
        print(f"{pct:>11.0f}%", *[f"{agree_rows[s][i]:>10.0%}" for s in strategies])

    print("\n[1] is the harsh full-KV metric; [2] is what CacheBlend's 'recover the quality' actually means.")
    print("Watch [2]-hkvd: the recompute % where it hits ~100% is the real 'recovers most' point.")
    print("HKVD should reach full agreement at a far smaller % than recency/random.")


if __name__ == "__main__":
    main()
