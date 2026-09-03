#!/usr/bin/env python3
"""Render the Results section of docs/experiments/speculative-test-execution.md from runs/summary.json and the production-trace files."""
import json, subprocess, re
subprocess.run(["python3", "analyze_replay.py"], capture_output=True)
S = json.load(open("runs/summary.json"))
T = "/tmp/claude-0/-home-user-dexa/86b25b67-2276-559a-a54a-f598498ce4d8/scratchpad/traces/"
lat = json.load(open(T + "tracelab_test_latency.json")); mdm = json.load(open(T + "tracelab_minDM.json"))
def f(x, d=2): return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
def pc(x): return "n/a" if x is None else f"{100*x:.0f}%"
rows = []
for g in ("swe-agent", "mini-swe-agent", "all"):
    v = S["by_framework"].get(g)
    if not v: continue
    A = v["hit_rate_by_launching_edit_kind"]; B = v["policy_B_run_created_file"]
    mod = A.get("modified", {}); cre = A.get("created", {})
    rows.append(f"| {g} | {v['sessions']} | {v['calls']} | {v['launches']} | {pc(v['hit_rate_per_launch'])} | {mod.get('launches',0)} | {pc(mod.get('hit_rate'))} | {B['created_edits_with_prediction']} | {pc(B['hit_rate'])} | {pc(v['hit_output_equal_normalized'])} | {f(v['hit_duration_s']['p50'])} / {f(v['hit_duration_s']['p90'])} |")
table1 = "\n".join(["| trajectories | sessions | tool calls | launches, any edit | hit rate, any edit | launches after modifying a file | hit rate | created-file predictions | hit rate, run the new file | speculative output equals real | hit run duration p50 / p90 (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"] + rows)
lat_rows = []
for k in ("claude:pytest", "claude:python", "claude:build-tool", "codex:pytest", "codex:python"):
    if k in lat:
        l = lat[k]; lat_rows.append(f"| {k.replace(':', ' ')} | {l['n']:,} | {f(l['p50'],1)} | {f(l['p90'],1)} | {f(l['p99'],0)} | {pc(l['share_over_5s'])} | {pc(l['time_share_over_5s'])} |")
table2 = "\n".join(["| production tool | calls | p50 (s) | p90 (s) | p99 (s) | share over 5 s | time in calls over 5 s |", "|---|---|---|---|---|---|---|"] + lat_rows)
sav_rows = []
for k in ("pytest", "python"):
    m = mdm[k]; e = m["expected_saving_per_hit"]
    sav_rows.append(f"| Claude Code {k} run | {f(m['mean_D'],1)} | {f(e['1.5'],1)} | {f(e['6.6'],1)} | {f(e['14.2'],1)} | {f(e['26.2'],1)} |")
table3 = "\n".join(["| hit on a ... | mean duration (s) | saved at model step 1.5 s | 6.6 s (p50) | 14.2 s (mean) | 26.2 s (p90) |", "|---|---|---|---|---|---|"] + sav_rows)
all_ = S["by_framework"]["all"]; A = all_["hit_rate_by_launching_edit_kind"]; B = all_["policy_B_run_created_file"]
results = f"""## Results

Replayed so far: **{all_['sessions']} sessions, {all_['calls']:,} tool calls** across {len(set(json.loads(l)['image'] for fn in __import__('glob').glob('runs/*.jsonl') for l in open(fn) if 'summary' in l))} repository images.

### 1. How predictable is the command after an edit?

{table1}

Two rules cover the post-edit step. **Rule A**, after a call that *modifies* an existing file, launch the most recent test-like command: hit rate {pc(A.get('modified',{}).get('hit_rate'))} over {A.get('modified',{}).get('launches',0)} launches. **Rule B**, after a call that *creates* a file, launch that file: hit rate {pc(B['hit_rate'])} over {B['created_edits_with_prediction']} predictions. Launching the old test after a file creation never hits ({A.get('created',{}).get('launches',0)} launches, {pc(A.get('created',{}).get('hit_rate'))}), which is why a single "rerun the last test" rule measures only {pc(all_['hit_rate_per_launch'])} across all edits. The `unknown` edit kinds are records from before the edit-kind field was added.

When a speculative run hits, its output matched the output of a real run on the same tree in {pc(all_['hit_output_equal_normalized'])} of cases after normalizing timings and stdout/stderr interleaving; every remaining mismatch inspected was ordering of interleaved streams.

### 2. How long are the runs being overlapped?

In these SWE-smith repositories the speculated runs are short (hit-run duration p50 {f(all_['hit_duration_s']['p50'])} s, p90 {f(all_['hit_duration_s']['p90'])} s), so the absolute saving inside the benchmark is small. The duration that matters is the production one. From the TraceLab release of real Claude Code and Codex sessions:

{table2}

### 3. What a live agent would save per hit

A hit saves `min(D, M)`: the test's duration `D`, capped by the model time `M` of the next step it overlaps. Taking `D` from the production Claude Code distribution above and `M` from TraceLab's per-step generation time:

{table3}

Per hit, a coding agent on today's model speeds saves about {f(mdm['pytest']['expected_saving_per_hit']['6.6'],0)} to {f(mdm['pytest']['expected_saving_per_hit']['14.2'],0)} seconds on a pytest rerun and {f(mdm['python']['expected_saving_per_hit']['6.6'],0)} to {f(mdm['python']['expected_saving_per_hit']['14.2'],0)} on a script rerun; at fast-inference model steps of 1.5 s the saving per hit collapses to about a second, because the overlap window is the model step. The lever pays in proportion to how slow the model is and how slow the tests are, and it is bounded by the number of post-edit reruns per task.
"""
doc = open("/home/user/dexa/docs/experiments/speculative-test-execution.md").read()
start = doc.index("## Results"); end = doc.index("## Caveats")
doc = doc[:start] + results + "\n" + doc[end:]
open("/home/user/dexa/docs/experiments/speculative-test-execution.md", "w").write(doc)
print("results rendered:", all_["sessions"], "sessions")
