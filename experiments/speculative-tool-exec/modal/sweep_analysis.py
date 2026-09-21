#!/usr/bin/env python3
"""Analyze a concurrency sweep: per level, task timing and per-step model latency from c<C>.jsonl, the engine's own
counters from the before/after metrics snapshots (c<C>.before.txt, c<C>.after.txt), and the KV-usage and running-
request gauges sampled during the level (metrics_samples.txt). Writes sweep.md."""
import glob, json, os, re, statistics as st, sys
from datetime import datetime, timezone


def scalar(txt, name):
    m = re.search(r'^vllm:%s\{[^}]*\} ([\d.e+]+)$' % name, txt, re.M)
    return float(m.group(1)) if m else 0.0


def p(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0


def main(run_dir):
    levels = sorted(int(re.search(r"c(\d+)\.jsonl$", f).group(1)) for f in glob.glob(os.path.join(run_dir, "c*.jsonl")))
    samples = []
    sp = os.path.join(run_dir, "metrics_samples.txt")
    if os.path.exists(sp):
        cur = None
        for line in open(sp):
            m = re.match(r"^=== (\S+)", line)
            if m:
                cur = {"t": datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))}; samples.append(cur); continue
            if cur is None: continue
            for key in ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting"):
                mm = re.match(r'^vllm:%s\{[^}]*\} ([\d.e+]+)' % key, line)
                if mm: cur[key] = float(mm.group(1))
    rows = []
    for C in levels:
        recs = [json.loads(l) for l in open(os.path.join(run_dir, f"c{C}.jsonl")) if l.strip()]
        ok = [r for r in recs if not r.get("error")]
        starts = [datetime.fromisoformat(r["started_at"]) for r in recs if r.get("started_at")]; ends = [datetime.fromisoformat(r["ended_at"]) for r in recs if r.get("ended_at")]
        span = (max(ends) - min(starts)).total_seconds() if starts and ends else 0
        loops = [r["agent_s"] - r.get("sandbox_boot_s", 0) for r in ok]
        steps = [s["model_s"] for r in ok for s in r["steps"] if "model_s" in s]
        stall = sum(max(0, x - 120) for x in steps)
        row = {"C": C, "tasks": len(recs), "errors": len(recs) - len(ok), "submitted": sum(1 for r in ok if r.get("submitted")),
               "loop_median": st.median(loops) if loops else 0, "loop_mean": st.mean(loops) if loops else 0,
               "step_median": st.median(steps) if steps else 0, "step_p90": p(steps, .9), "calls": len(steps), "stall_s": stall,
               "span": span, "tph_span": len(ok) * 3600 / span if span else 0, "tph_steady": len(ok) * 3600 / (sum(loops) / C) if loops else 0,
               "tph_steady_stallfree": len(ok) * 3600 / ((sum(loops) - stall) / C) if loops else 0,
               "tool_median": st.median(r["tool_s"] for r in ok) if ok else 0, "steps_mean": st.mean(len(r["steps"]) for r in ok) if ok else 0}
        b = os.path.join(run_dir, f"c{C}.before.txt"); a = os.path.join(run_dir, f"c{C}.after.txt")
        if os.path.exists(b) and os.path.exists(a):
            tb, ta = open(b).read(), open(a).read()
            d = lambda n: scalar(ta, n) - scalar(tb, n)
            q = d("prefix_cache_queries_total"); h = d("prefix_cache_hits_total"); n = d("request_prompt_tokens_count") or 1
            row.update({"prefix_hit": h / q if q else 0, "uncached_per_call": (q - h) / n, "preemptions": d("num_preemptions_total"),
                        "prefill_s": d("request_prefill_time_seconds_sum"), "decode_s": d("request_decode_time_seconds_sum"),
                        "ttft_ms": d("time_to_first_token_seconds_sum") / n * 1000, "tpot_ms": d("time_per_output_token_seconds_sum") / max(1, d("time_per_output_token_seconds_count")) * 1000,
                        "prompt_tok": d("prompt_tokens_total"), "gen_tok": d("generation_tokens_total"), "requests": n,
                        "gen_tok_per_s": d("generation_tokens_total") / span if span else 0})
        if samples and starts:
            t0, t1 = min(starts), max(ends)
            inwin = [x for x in samples if t0 <= x["t"] <= t1 and "kv_cache_usage_perc" in x]
            if inwin:
                row.update({"kv_mean": st.mean(x["kv_cache_usage_perc"] for x in inwin) * 100, "kv_max": max(x["kv_cache_usage_perc"] for x in inwin) * 100,
                            "running_mean": st.mean(x.get("num_requests_running", 0) for x in inwin), "running_max": max(x.get("num_requests_running", 0) for x in inwin),
                            "waiting_max": max(x.get("num_requests_waiting", 0) for x in inwin), "samples": len(inwin)})
        rows.append(row)
    out = [f"# Concurrency sweep `{os.path.basename(run_dir)}`\n", "Vanilla arm, one H100, levels run one after the other on the same server. Engine counters are deltas over each level; gauges are sampled every 15 s during it.\n"]
    out.append("| sessions in flight | task runs | errors | submitted | loop median (s) | loop mean (s) | model call median (s) | p90 (s) | stall time (s) | tool median (s) | tasks per GPU-hour, level span | steady state | steady state, stall-free |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(f"| {r['C']} | {r['tasks']} | {r['errors']} | {r['submitted']} | {r['loop_median']:.0f} | {r['loop_mean']:.0f} | {r['step_median']:.2f} | {r['step_p90']:.2f} | {r['stall_s']:.0f} | {r['tool_median']:.0f} | {r['tph_span']:.0f} | {r['tph_steady']:.0f} | {r['tph_steady_stallfree']:.0f} |")
    out.append("\n| sessions in flight | model calls | prefix-cache hit rate | uncached prompt tokens per call | preemptions | prefill time (s) | decode time (s) | TTFT mean (ms) | TPOT mean (ms) | generated tokens per s | KV usage mean / max (%) | running requests mean / max | waiting max |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "prefix_hit" in r:
            out.append(f"| {r['C']} | {r['requests']:.0f} | {r['prefix_hit']:.3f} | {r['uncached_per_call']:.0f} | {r['preemptions']:.0f} | {r['prefill_s']:.0f} | {r['decode_s']:.0f} | {r['ttft_ms']:.0f} | {r['tpot_ms']:.1f} | {r['gen_tok_per_s']:.0f} | {r.get('kv_mean', 0):.0f} / {r.get('kv_max', 0):.0f} | {r.get('running_mean', 0):.1f} / {r.get('running_max', 0):.0f} | {r.get('waiting_max', 0):.0f} |")
    text = "\n".join(out) + "\n"
    open(os.path.join(run_dir, "sweep.md"), "w").write(text); json.dump(rows, open(os.path.join(run_dir, "sweep.json"), "w"), indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
