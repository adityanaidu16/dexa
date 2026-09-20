#!/usr/bin/env python3
"""Summarize one run directory of live_agent_modal.py outputs (one .jsonl per arm) as a markdown table.

Throughput is tasks per GPU-hour with the one server dedicated to the arm: n_tasks * 3600 / arm_wall, where arm_wall is
the span from the arm's first task start to its last task end. Resolve rate uses the official SWE-bench grade.
"""
import glob, json, os, statistics as st, sys
from datetime import datetime


def load(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def med(xs):
    return st.median(xs) if xs else 0.0


def summarize(run_dir):
    arms = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        arm = os.path.splitext(os.path.basename(path))[0]
        recs = load(path)
        ok = [r for r in recs if not r.get("error")]
        starts = [datetime.fromisoformat(r["started_at"]) for r in recs if r.get("started_at")]
        ends = [datetime.fromisoformat(r["ended_at"]) for r in recs if r.get("ended_at")]
        span = (max(ends) - min(starts)).total_seconds() if starts and ends else 0.0
        launches = sum(r.get("launches", 0) for r in ok); hits = sum(r.get("hits", 0) for r in ok); misses = sum(r.get("misses", 0) for r in ok)
        arms[arm] = {
            "tasks": len(recs), "errors": len(recs) - len(ok),
            "resolved": sum(1 for r in ok if (r.get("grade") or {}).get("resolved")),
            "submitted": sum(1 for r in ok if r.get("submitted")),
            "wall_mean_s": st.mean([r["wall_s"] for r in ok]) if ok else 0, "wall_median_s": med([r["wall_s"] for r in ok]),
            "agent_mean_s": st.mean([r.get("agent_s", r["wall_s"]) for r in ok]) if ok else 0,
            "model_mean_s": st.mean([r["model_s"] for r in ok]) if ok else 0, "tool_mean_s": st.mean([r["tool_s"] for r in ok]) if ok else 0,
            "boot_mean_s": st.mean([r.get("sandbox_boot_s", 0) for r in ok]) if ok else 0,
            "steps_mean": st.mean([len(r["steps"]) for r in ok]) if ok else 0,
            "prompt_tokens_mean": st.mean([r["tokens"].get("prompt", 0) for r in ok]) if ok else 0,
            "completion_tokens_mean": st.mean([r["tokens"].get("completion", 0) for r in ok]) if ok else 0,
            "launches": launches, "hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses),
            "saved_s_mean": st.mean([r.get("saved_s", 0) for r in ok]) if ok else 0, "wasted_s_mean": st.mean([r.get("wasted_s", 0) for r in ok]) if ok else 0,
            "concurrency": max((r.get("concurrency", 1) for r in recs), default=1),
            "arm_span_s": span, "tasks_per_gpu_hour": (len(recs) * 3600 / span) if span else 0,
            "agent_sum_s": sum(r.get("agent_s", r["wall_s"]) for r in ok),
        }
        conc = arms[arm]["concurrency"]
        arms[arm]["tasks_per_gpu_hour_steady"] = (len(ok) * 3600 / (arms[arm]["agent_sum_s"] / conc)) if arms[arm]["agent_sum_s"] else 0
    # interleaved runs: the two arms' spans overlap almost entirely
    spans = {}
    for arm in arms:
        recs = load(os.path.join(run_dir, f"{arm}.jsonl"))
        starts = [datetime.fromisoformat(r["started_at"]) for r in recs if r.get("started_at")]; ends = [datetime.fromisoformat(r["ended_at"]) for r in recs if r.get("ended_at")]
        if starts and ends: spans[arm] = (min(starts), max(ends))
    if "off" in spans and "harness" in spans:
        a, b = spans["off"], spans["harness"]
        overlap = (min(a[1], b[1]) - max(a[0], b[0])).total_seconds()
        shorter = min((a[1] - a[0]).total_seconds(), (b[1] - b[0]).total_seconds())
        if shorter > 0 and overlap / shorter > 0.5:
            for arm in arms: arms[arm]["interleaved"] = True
    return arms


def render(arms):
    rows = [("tasks", "tasks", "d"), ("errors", "errors", "d"), ("resolved", "resolved (official grade)", "d"), ("submitted", "submitted", "d"),
            ("concurrency", "concurrency", "d"), ("tasks_per_gpu_hour", "tasks per GPU-hour (arm span)", ".2f"), ("tasks_per_gpu_hour_steady", "tasks per GPU-hour (steady state, no tail)", ".2f"), ("arm_span_s", "arm span (s)", ".0f"),
            ("wall_mean_s", "wall per task, mean (s)", ".0f"), ("wall_median_s", "wall per task, median (s)", ".0f"), ("agent_mean_s", "agent loop per task, mean (s)", ".0f"),
            ("model_mean_s", "model time per task (s)", ".0f"), ("tool_mean_s", "tool time per task (s)", ".0f"), ("boot_mean_s", "sandbox boot (s)", ".1f"),
            ("steps_mean", "steps per task", ".1f"), ("prompt_tokens_mean", "prompt tokens per task", ".0f"), ("completion_tokens_mean", "completion tokens per task", ".0f"),
            ("launches", "speculative launches", "d"), ("hits", "hits", "d"), ("misses", "misses", "d"), ("hit_rate", "hit rate", ".2f"),
            ("saved_s_mean", "saved per task (s)", ".1f"), ("wasted_s_mean", "wasted per task (s)", ".1f")]
    names = list(arms)
    out = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, fmt in rows:
        out.append(f"| {label} | " + " | ".join(format(arms[n][key], fmt) for n in names) + " |")
    if "off" in arms and "harness" in arms and arms["off"]["tasks_per_gpu_hour"]:
        ratio = arms["harness"]["tasks_per_gpu_hour"] / arms["off"]["tasks_per_gpu_hour"]
        ratio2 = (arms["harness"]["tasks_per_gpu_hour_steady"] / arms["off"]["tasks_per_gpu_hour_steady"]) if arms["off"]["tasks_per_gpu_hour_steady"] else 0
        out.append("")
        if arms.get("off", {}).get("interleaved"):
            out.append(f"interleaved run: both arms shared the server for the whole span, so arm-span throughput is not meaningful and the absolute steady-state figures assume each arm had the server alone; the ratio is the number to read. harness / off ratio of summed agent loop: {ratio2:.3f}  (gate: >= 1.30 continue, < 1.15 kill; judged only at equal resolve rate; see analysis.md for the paired, boot-corrected version)")
        else:
            out.append(f"harness / off throughput ratio: {ratio:.3f} by arm span, {ratio2:.3f} steady state  (gate: >= 1.30 continue, < 1.15 kill; judged only at equal resolve rate)")
    return "\n".join(out)


if __name__ == "__main__":
    run_dir = sys.argv[1]
    arms = summarize(run_dir)
    json.dump(arms, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    print(render(arms))
    if arms and all(a["tasks"] and a["errors"] == a["tasks"] for a in arms.values()):
        print("every task errored; treating the run as failed", file=sys.stderr)
        sys.exit(2)
