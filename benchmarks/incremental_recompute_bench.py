"""Phase 1 benchmark: tokens reprocessed per mutating turn — incremental recompute
vs full re-prefill — on a simulated long-horizon agent loop.

The README's Phase-1 gate is "order-of-magnitude reduction in tokens reprocessed
per mutating turn vs full re-prefill on an agent loop." This measures exactly that.
The **tokens-reprocessed** counts are hardware-independent and exact (from the
recompute planner); wall-time uses a real model (tiny-random Llama by default, CPU)
so it is *indicative* — the token ratio is the number that transfers to an 8B/GPU.

Workload. A coding-agent context: a stable [system][tools] header, a few [repo doc]
segments, then a growing tail of [turn]/[tool_result] segments. Each step mutates
the context the way an agent actually does — usually appending a tool result or
editing the most recent one (edit near the end → tiny recompute), occasionally
editing a mid-context doc (edit earlier → larger recompute). Incremental recompute
reuses the unchanged prefix; full re-prefill reprocesses everything every turn.

  python benchmarks/incremental_recompute_bench.py --steps 20 --model hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from __future__ import annotations

import argparse
import time

from dexa.segment import Segment, SegmentedContext, plan_incremental


def _backend(model, device):
    import torch
    from dexa.engine.hf_backend import HFBackend
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(f"loading {model} on {device}/{dtype} ...", flush=True)
    return HFBackend(model_name=model, device=device, dtype=dtype)


def _seg(be, name, text, role):
    return Segment(name=name, token_ids=tuple(be.tokenize(text)), role=role)


_LOREM = ("The quick brown fox jumps over the lazy dog. "
          "A journey of a thousand miles begins with a single step. ")


def _build_initial(be):
    return SegmentedContext([
        _seg(be, "system", "You are an autonomous coding agent working in a large repository. "
             "Follow instructions carefully and use tools to inspect and edit files.", "system"),
        _seg(be, "tools", "Available tools: read_file, write_file, run_tests, grep, list_dir. "
             "Call them with JSON arguments.", "tools"),
        _seg(be, "doc0", "File util.py: " + _LOREM * 3, "doc"),
        _seg(be, "doc1", "File main.py: " + _LOREM * 3, "doc"),
    ])


def _mutate(be, ctx, step):
    """Return the next SegmentedContext, emulating an agent turn.

    Cycle: append tool_result, edit that tool_result, append turn, and every 5th
    step edit a mid-context doc (the expensive-for-full-prefill case)."""
    segs = list(ctx.segments)
    kind = step % 5
    if kind == 4:
        # mid-context edit: rewrite doc1 (forces recompute from there on).
        for i, s in enumerate(segs):
            if s.name == "doc1":
                segs[i] = _seg(be, "doc1", f"File main.py (rev {step}): " + _LOREM * 3, "doc")
                break
        return SegmentedContext(segs), "edit_mid_doc"
    if kind in (0, 2):
        segs.append(_seg(be, f"tool_result{step}",
                         f"tool_result step {step}: " + _LOREM * 2, "tool_result"))
        return SegmentedContext(segs), "append_tool_result"
    if kind == 1 and segs:
        # edit the most recent tool_result (near the end -> cheap incremental).
        segs[-1] = _seg(be, segs[-1].name, f"tool_result (retry {step}): " + _LOREM * 2, "tool_result")
        return SegmentedContext(segs), "edit_recent_result"
    segs.append(_seg(be, f"turn{step}", f"User: what changed at step {step}?", "query"))
    return SegmentedContext(segs), "append_turn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--timed", action="store_true", help="also run the model (wall-time; slow)")
    args = ap.parse_args()

    be = _backend(args.model, args.device) if args.timed else None

    class _Toks:
        """Token-only stand-in so the planner runs without loading a model."""
        def tokenize(self, text):
            return list(range(abs(hash(text)) % 1000, abs(hash(text)) % 1000 + max(1, len(text) // 4)))
    tk = be if be is not None else _Toks()

    ctx = _build_initial(tk)
    prev_kv = be.prefill(ctx.token_ids) if be is not None else None

    tot_inc, tot_full = 0, 0
    inc_time, full_time = 0.0, 0.0
    print(f"\n{'step':>4} {'action':<20}{'ctx toks':>9}{'full re-pf':>11}{'incr':>7}{'incr %':>8}")
    for step in range(1, args.steps + 1):
        new_ctx, action = _mutate(tk, ctx, step)
        plan = plan_incremental(ctx, new_ctx, mode="exact")
        s = plan.savings()
        inc = s["recomputed_tokens_exact"]
        full = new_ctx.n_tokens
        tot_inc += inc
        tot_full += full

        if be is not None:
            t0 = time.perf_counter()
            kv, _ = be.recompute_incremental(prev_kv, ctx, new_ctx)
            inc_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            _ = be.prefill(new_ctx.token_ids)
            full_time += time.perf_counter() - t0
            prev_kv = kv

        print(f"{step:>4} {action:<20}{full:>9}{full:>11}{inc:>7}{100*inc/full:>7.0f}%")
        ctx = new_ctx

    print(f"\ntotals over {args.steps} mutating turns:")
    print(f"  tokens reprocessed  full re-prefill : {tot_full}")
    print(f"  tokens reprocessed  incremental     : {tot_inc}")
    print(f"  reduction                           : {tot_full/max(tot_inc,1):.1f}x fewer tokens reprocessed")
    if be is not None:
        print(f"  wall-time (indicative, {args.model.split('/')[-1]}): "
              f"full {full_time*1e3:.0f}ms vs incremental {inc_time*1e3:.0f}ms "
              f"({full_time/max(inc_time,1e-9):.1f}x)")


if __name__ == "__main__":
    main()
