"""On-device incremental recompute — the wall-time payoff (Phase 1, path B).

The first GPU run showed the token savings are real (4.4x fewer) but the numpy
round-trip made incremental *slower* than a full re-prefill. This benchmark keeps
the KV resident on the GPU (TorchKVSession) and times three paths per mutating
turn on the same agent loop:

  full        — re-prefill the whole context every turn (no reuse)
  numpy-incr  — HFBackend.recompute_incremental (reuses prefix, but via CPU numpy)
  ondevice    — TorchKVSession (reuses prefix, KV never leaves the GPU)

The claim under test: `ondevice` beats `full` in wall-time (token savings finally
realized), while `numpy-incr` does not. Ends with a correctness check: the
on-device session's greedy continuation matches a full re-prefill.

  python benchmarks/ondevice_incremental_bench.py --steps 20 --model unsloth/Llama-3.1-8B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import time

from dexa.segment import Segment, SegmentedContext


def _backend(model, device):
    import torch
    from dexa.engine.hf_backend import HFBackend
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(f"loading {model} on {device}/{dtype} ...", flush=True)
    return HFBackend(model_name=model, device=device, dtype=dtype), device


def _seg(be, name, text, role):
    return Segment(name, tuple(be.tokenize(text)), role)


_L = "The quick brown fox jumps over the lazy dog near the river bank at dawn. "


def _initial(be):
    return SegmentedContext([
        _seg(be, "system", "You are an autonomous coding agent in a large repository. "
             "Use tools to inspect and edit files.", "system"),
        _seg(be, "tools", "Tools: read_file, write_file, run_tests, grep, list_dir.", "tools"),
        _seg(be, "doc0", "File util.py: " + _L * 3, "doc"),
        _seg(be, "doc1", "File main.py: " + _L * 3, "doc"),
    ])


def _mutate(be, ctx, step):
    segs = list(ctx.segments)
    kind = step % 5
    if kind == 4:
        for i, s in enumerate(segs):
            if s.name == "doc1":
                segs[i] = _seg(be, "doc1", f"File main.py (rev {step}): " + _L * 3, "doc")
                break
        return SegmentedContext(segs)
    if kind in (0, 2):
        segs.append(_seg(be, f"tool_result{step}", f"tool_result {step}: " + _L * 2, "tool_result"))
        return SegmentedContext(segs)
    if kind == 1:
        segs[-1] = _seg(be, segs[-1].name, f"tool_result (retry {step}): " + _L * 2, "tool_result")
        return SegmentedContext(segs)
    segs.append(_seg(be, f"turn{step}", f"User: what changed at step {step}?", "query"))
    return SegmentedContext(segs)


def _scaling_sweep(be, torch, device, lengths, delta):
    """One representative mutation (append ~delta tokens) at a range of context
    sizes L, timing on-device append (O(delta)) vs full re-prefill (O(L)). Shows the
    speedup widening with L — the long-horizon signature.

    Not an agent loop: each row is an independent 'hold L tokens, mutate once'
    measurement, so the trend is purely context size, not accumulation."""
    from dexa.engine.torch_session import TorchKVSession
    cuda = device == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def timed(fn):
        sync(); t0 = time.perf_counter(); r = fn(); sync()
        return r, time.perf_counter() - t0

    # a long, deterministic token pool to slice contexts of exact length from.
    pool = be.tokenize(_L * 4000)
    tail = pool[:delta]     # a real delta-token append
    print(f"\nscaling sweep — one {delta}-token append at each context length "
          f"(model on {device}); on-device append vs full re-prefill:\n")
    print(f"{'ctx tokens':>11} {'full ms':>10} {'ondev ms':>10} {'speedup':>9} "
          f"{'tok full':>9} {'tok ondev':>10}")
    rows = []
    for L in lengths:
        if L > len(pool):
            print(f"{L:>11}  (skipped: pool too small)"); continue
        base = pool[:L]
        # warm the kernels at this shape, then measure.
        sess = TorchKVSession(be, base)
        _ = be.prefill(base + tail)
        _, dt_full = timed(lambda: be.prefill(base + tail))
        _, dt_dev = timed(lambda: sess.append(tail))
        speedup = dt_full / max(dt_dev, 1e-9)
        rows.append((L, dt_full * 1e3, dt_dev * 1e3, speedup))
        print(f"{L:>11} {dt_full*1e3:>10.1f} {dt_dev*1e3:>10.1f} {speedup:>8.1f}x "
              f"{L+len(tail):>9} {len(tail):>10}")
        del sess
        if cuda:
            torch.cuda.empty_cache()

    print("\non-device append cost is ~flat (O(delta)); full re-prefill grows with context")
    print("(O(L)) -> the speedup climbs with context length. At agentic 10k-100k-token")
    print("contexts this is the difference between a re-prefill stall and an instant turn.")
    if len(rows) >= 2:
        (L0, _, _, s0), (L1, _, _, s1) = rows[0], rows[-1]
        print(f"\nspeedup {s0:.1f}x @ {L0} tok  ->  {s1:.1f}x @ {L1} tok "
              f"({s1/max(s0,1e-9):.1f}x wider over {L1//max(L0,1)}x more context).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--scaling", action="store_true",
                    help="run the context-length scaling sweep instead of the agent loop")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384",
                    help="context lengths for --scaling")
    ap.add_argument("--delta", type=int, default=48, help="mutation size for --scaling")
    args = ap.parse_args()

    import torch
    from dexa.engine.torch_session import TorchKVSession
    be, device = _backend(args.model, args.device)
    cuda = device == "cuda"

    if args.scaling:
        _scaling_sweep(be, torch, device, [int(x) for x in args.lengths.split(",")], args.delta)
        return

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def timed(fn):
        sync(); t0 = time.perf_counter(); r = fn(); sync()
        return r, time.perf_counter() - t0

    ctx = _initial(be)
    # separate state per path.
    prev_kv = be.prefill(ctx.token_ids)              # numpy-incr path
    sess = TorchKVSession(be, ctx.token_ids)         # ondevice path

    # warmup (kernel autotune / allocator) so the first timed step isn't penalized.
    _ = be.prefill(ctx.token_ids)

    t_full = t_numpy = t_dev = 0.0
    tok_full = tok_inc = 0
    print(f"\n{'step':>4} {'ctxTok':>7} {'full ms':>9} {'numpy ms':>9} {'ondev ms':>9}")
    for step in range(1, args.steps + 1):
        new_ctx = _mutate(be, ctx, step)

        _, dt_full = timed(lambda: be.prefill(new_ctx.token_ids))
        (kv2, _st), dt_numpy = timed(lambda: be.recompute_incremental(prev_kv, ctx, new_ctx))
        _st2, dt_dev = timed(lambda: sess.apply(ctx, new_ctx))

        prev_kv = kv2
        t_full += dt_full; t_numpy += dt_numpy; t_dev += dt_dev
        tok_full += new_ctx.n_tokens; tok_inc += _st2["recomputed_tokens"]
        print(f"{step:>4} {new_ctx.n_tokens:>7} {dt_full*1e3:>9.1f} {dt_numpy*1e3:>9.1f} {dt_dev*1e3:>9.1f}")
        ctx = new_ctx

    print(f"\ntotals over {args.steps} mutating turns ({device}):")
    print(f"  tokens reprocessed : full {tok_full}  |  incremental {tok_inc}  "
          f"({tok_full/max(tok_inc,1):.1f}x fewer)")
    print(f"  wall-time full re-prefill : {t_full*1e3:8.0f} ms")
    print(f"  wall-time numpy-incr      : {t_numpy*1e3:8.0f} ms  ({t_full/max(t_numpy,1e-9):.2f}x vs full)")
    print(f"  wall-time ONDEVICE-incr   : {t_dev*1e3:8.0f} ms  ({t_full/max(t_dev,1e-9):.2f}x vs full)")

    # correctness: on-device session behaves like a full re-prefill of the final ctx.
    full = be.prefill(ctx.token_ids)
    ok = sess.greedy(8) == be.generate(full, [], max_new_tokens=8)
    print(f"\n  on-device final state matches full re-prefill (greedy): {ok}")


if __name__ == "__main__":
    main()
