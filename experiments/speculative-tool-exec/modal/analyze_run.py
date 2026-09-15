#!/usr/bin/env python3
"""Paired analysis of one run directory (off.jsonl + harness.jsonl): per-task table, boot-corrected throughput,
speculation accounting, per-step model latency by context depth, and stall detection. Writes analysis.md."""
import json, os, statistics as st, sys


def load(p):
    return {r["instance_id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def loop(r):
    return r.get("agent_s", r["wall_s"]) - r.get("sandbox_boot_s", 0)


def p(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]


def main(run_dir):
    off = load(os.path.join(run_dir, "off.jsonl")); har = load(os.path.join(run_dir, "harness.jsonl"))
    out = []
    W = out.append
    errs = [(i, r["error"][:160]) for i, r in har.items() if r.get("error")] + [(i, r["error"][:160]) for i, r in off.items() if r.get("error")]
    W(f"# Paired analysis of `{os.path.basename(run_dir)}`\n")
    W(f"Tasks: {len(off)} off, {len(har)} harness. Errored records: {len(errs)}.\n")
    if errs:
        W("Errors (excluded from paired statistics; counted as unresolved):\n")
        for i, e in errs: W(f"- {i}: {e}")
        W("")
    pairs = [(off[i], har[i]) for i in off if i in har and not har[i].get("error") and not off[i].get("error")]
    n = len(pairs)
    res_o = sum(1 for o, h in pairs if (o.get("grade") or {}).get("resolved")); res_h = sum(1 for o, h in pairs if (h.get("grade") or {}).get("resolved"))
    W(f"## Paired tasks (n={n})\n")
    W("| metric | off | harness |\n|---|---|---|")
    W(f"| resolved (official grade) | {res_o} | {res_h} |")
    W(f"| agent loop minus sandbox boot, median (s) | {st.median(loop(o) for o, h in pairs):.0f} | {st.median(loop(h) for o, h in pairs):.0f} |")
    W(f"| agent loop minus sandbox boot, mean (s) | {st.mean(loop(o) for o, h in pairs):.0f} | {st.mean(loop(h) for o, h in pairs):.0f} |")
    W(f"| sandbox boot, mean (s) | {st.mean(o.get('sandbox_boot_s', 0) for o, h in pairs):.1f} | {st.mean(h.get('sandbox_boot_s', 0) for o, h in pairs):.1f} |")
    W(f"| model time, median (s) | {st.median(o['model_s'] for o, h in pairs):.0f} | {st.median(h['model_s'] for o, h in pairs):.0f} |")
    W(f"| tool time, median (s) | {st.median(o['tool_s'] for o, h in pairs):.0f} | {st.median(h['tool_s'] for o, h in pairs):.0f} |")
    W(f"| steps, mean | {st.mean(len(o['steps']) for o, h in pairs):.1f} | {st.mean(len(h['steps']) for o, h in pairs):.1f} |")
    ratios = [loop(h) / loop(o) for o, h in pairs]
    so = sum(loop(o) for o, h in pairs); sh = sum(loop(h) for o, h in pairs)
    conc = max(r.get("concurrency", 1) for r in off.values())
    W(f"\nPer-task ratio harness/off of the boot-corrected loop: median {st.median(ratios):.3f}, IQR {p(ratios, .25):.2f} to {p(ratios, .75):.2f}.")
    W(f"Tasks per GPU-hour on the paired tasks at concurrency {conc}, boot excluded: off {n * 3600 / (so / conc):.0f}, harness {n * 3600 / (sh / conc):.0f}, ratio {so / sh:.3f}.")
    k = 3; so_t = sum(sorted(loop(o) for o, h in pairs)[:-k]); sh_t = sum(sorted(loop(h) for o, h in pairs)[:-k])
    W(f"Dropping the {k} slowest tasks in each arm: ratio {so_t / sh_t:.3f}.\n")
    # speculation accounting
    hits = [e for o, h in pairs for e in h.get("spec_events", []) if e["kind"] == "hit"]
    misses = [e for o, h in pairs for e in h.get("spec_events", []) if e["kind"] == "miss"]
    launches = sum(h.get("launches", 0) for o, h in pairs)
    W("## Speculation accounting (harness arm, paired tasks)\n")
    W(f"- launches {launches} ({launches / n:.1f} per task), hits {len(hits)}, misses {len(misses)}, hit rate {len(hits) / max(1, len(hits) + len(misses)):.2f}")
    for rule in "AB":
        hr = [e for e in hits if e["rule"] == rule]; mr = [e for e in misses if e["rule"] == rule]
        W(f"- rule {rule}: {len(hr)} hits, {len(mr)} misses, hit rate {len(hr) / max(1, len(hr) + len(mr)):.2f}")
    if hits:
        W(f"- saved per hit: median {st.median(e['saved_s'] for e in hits):.2f} s, mean {st.mean(e['saved_s'] for e in hits):.2f} s, max {max(e['saved_s'] for e in hits):.1f} s; speculative command duration median {st.median(e['spec_duration_s'] for e in hits):.2f} s")
    W(f"- saved per task {sum(e['saved_s'] for e in hits) / n:.2f} s; killed-run time per task {sum(e['wasted_s'] for e in misses) / n:.2f} s (background, not wall clock); launch/poll/kill round trips are extra sandbox execs of roughly 0.3 to 0.5 s each, not itemized")
    W(f"- ceiling: if every launch had hit with zero overhead, saving would be at most {sum(e.get('spec_duration_s', 0) for e in hits) / n + sum(e.get('wasted_s', 0) for e in misses) / n:.1f} s per task against a median loop of {st.median(loop(o) for o, h in pairs):.0f} s\n")
    # model latency by depth and stalls
    W("## Model latency by context depth (both arms pooled)\n")
    by = {}
    for r in [o for o, h in pairs] + [h for o, h in pairs]:
        for s in r["steps"]:
            if "model_s" in s: by.setdefault(s["step"] // 10 * 10, []).append(s["model_s"])
    W("| steps | calls | median (s) | p90 (s) | max (s) |\n|---|---|---|---|---|")
    for kk in sorted(by):
        W(f"| {kk} to {kk + 9} | {len(by[kk])} | {st.median(by[kk]):.2f} | {p(by[kk], .9):.2f} | {max(by[kk]):.0f} |")
    stalls = [(r['instance_id'], r['arm'], s['step'], s['model_s']) for r in list(off.values()) + list(har.values()) for s in r['steps'] if s.get('model_s', 0) > 120]
    W(f"\nStalls (single model calls over 120 s): {len(stalls)}, total {sum(s[3] for s in stalls):.0f} s, across {len(set(s[0] + s[1] for s in stalls))} task runs: " + ", ".join(f"{i} ({a}, step {st_}, {m:.0f} s)" for i, a, st_, m in stalls))
    to = sum(r["tokens"].get("prompt", 0) for r in off.values()); tc = sum(r["tokens"].get("completion", 0) for r in off.values())
    W(f"\nOff arm token totals: {to / 1e6:.1f}M prompt, {tc / 1e3:.0f}k completion, ratio {to / max(1, tc):.0f}:1.\n")
    # per-task table
    W("## Per-task table\n")
    W("| instance | off loop (s) | off model | off tool | off steps | off resolved | harness loop (s) | model | tool | steps | resolved | hits | misses | saved (s) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i in sorted(off, key=lambda i: -(har.get(i, {}).get("wall_s", 0))):
        o = off[i]; h = har.get(i, {})
        if h.get("error"):
            W(f"| {i} | {loop(o):.0f} | {o['model_s']:.0f} | {o['tool_s']:.0f} | {len(o['steps'])} | {(o.get('grade') or {}).get('resolved')} | error | | | | | | | |"); continue
        W(f"| {i} | {loop(o):.0f} | {o['model_s']:.0f} | {o['tool_s']:.0f} | {len(o['steps'])} | {(o.get('grade') or {}).get('resolved')} | {loop(h):.0f} | {h['model_s']:.0f} | {h['tool_s']:.0f} | {len(h['steps'])} | {(h.get('grade') or {}).get('resolved')} | {h.get('hits', 0)} | {h.get('misses', 0)} | {h.get('saved_s', 0):.1f} |")
    text = "\n".join(out) + "\n"
    open(os.path.join(run_dir, "analysis.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
