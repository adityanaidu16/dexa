#!/usr/bin/env python3
"""Per-step timing decomposition of the TraceLab coding-agent trace (Claude Code + Codex).
Outputs JSON with: step wall-clock split (generation vs tool), tool latency by tool identity,
idle-window distribution for tool-triggered steps, and how well tool identity predicts a long wait."""
import gzip, json, sys, statistics as st
from datetime import datetime
from collections import defaultdict, Counter

def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs)-1)*p
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

LONG_S = 5.0
rounds_by_session = defaultdict(list)
n = 0
with gzip.open("syfi_coding_trace.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line); n += 1
        ev = r.get("timing_events") or []
        if not ev: continue
        t0 = ts(ev[0]["timestamp"])
        asst = [ts(e["timestamp"]) for e in ev if e["event_type"] in ("reasoning","text","tool_call")]
        tools = r.get("tools") or []
        rounds_by_session[r["session_id"]].append({
            "provider": r["provider"], "idx": r["round_index"], "t0": t0,
            "t_asst": max(asst) if asst else None,
            "first_in": r.get("first_input_event_type"),
            "tools": [(t.get("tool_name"), t.get("tool_wall_latency_ms"), (t.get("executables") or [None])[0],
                       t.get("result_chars"), t.get("is_error"), ts(t["result_at"]) if t.get("result_at") else None) for t in tools],
            "out_tok": r.get("output_tokens"), "append_tok": r.get("newly_append_tokens"),
        })
print("rounds", n, "sessions", len(rounds_by_session), file=sys.stderr)

# --- per-step decomposition on tool-triggered steps that issued tools
share = []            # tool share of step wall-clock
gen_ms, tool_par_ms, tool_ser_ms = [], [], []
idle_gap = {"tool": [], "user": []}      # seconds between last assistant event and next round start
tot = Counter()
by_prov = defaultdict(lambda: Counter())
lat = defaultdict(list)                  # (provider, tool, exe) -> latencies s
lat_tool = defaultdict(list)             # (provider, tool) -> latencies
calls_total = 0
for sid, rs in rounds_by_session.items():
    rs.sort(key=lambda r: r["idx"])
    for i, r in enumerate(rs):
        if r["t_asst"] is None: continue
        g = (r["t_asst"] - r["t0"]) * 1000.0
        if g < 0 or g > 3600e3: continue
        walls = [w for (_, w, *_ ) in r["tools"] if w is not None and w >= 0]
        for (name, w, exe, rc, err, rat) in r["tools"]:
            if w is None or w < 0: continue
            calls_total += 1
            lat[(r["provider"], name, exe)].append(w/1000.0)
            lat_tool[(r["provider"], name)].append(w/1000.0)
        if walls and (g + max(walls)) > 0:
            tp = max(walls); tsr = sum(walls)
            gen_ms.append(g); tool_par_ms.append(tp); tool_ser_ms.append(tsr)
            share.append(tp / (g + tp))
            tot["gen_ms"] += g; tot["tool_par_ms"] += tp; tot["tool_ser_ms"] += tsr
            by_prov[r["provider"]]["gen_ms"] += g; by_prov[r["provider"]]["tool_par_ms"] += tp; by_prov[r["provider"]]["steps"] += 1
        # idle window until the next round of the same session
        if i + 1 < len(rs):
            nxt = rs[i+1]
            gap = nxt["t0"] - r["t_asst"]
            if 0 <= gap <= 24*3600:
                key = "tool" if nxt["first_in"] == "tool_result" else "user"
                idle_gap[key].append(gap)

def summ(xs):
    return {"n": len(xs), "p50": pct(xs,.5), "p90": pct(xs,.9), "p99": pct(xs,.99), "mean": (sum(xs)/len(xs)) if xs else None}

out = {"rounds": n, "sessions": len(rounds_by_session), "tool_calls_with_latency": calls_total,
       "steps_with_tools": len(share),
       "step_generation_ms": summ(gen_ms), "step_tool_parallel_ms": summ(tool_par_ms), "step_tool_serial_ms": summ(tool_ser_ms),
       "tool_share_of_step_wallclock": {"per_step_median": pct(share,.5), "per_step_mean": sum(share)/len(share),
                                        "time_weighted": tot["tool_par_ms"]/(tot["gen_ms"]+tot["tool_par_ms"]),
                                        "steps_where_tool_exceeds_generation": sum(1 for s in share if s > 0.5)/len(share)},
       "by_provider": {p: {"steps": c["steps"], "time_weighted_tool_share": c["tool_par_ms"]/(c["gen_ms"]+c["tool_par_ms"]),
                           "mean_gen_s": c["gen_ms"]/c["steps"]/1000, "mean_tool_s": c["tool_par_ms"]/c["steps"]/1000} for p, c in by_prov.items()},
       "idle_window_s": {k: summ(v) for k, v in idle_gap.items()},
       "idle_window_tool_triggered_share_above": {str(x): sum(1 for g in idle_gap["tool"] if g > x)/len(idle_gap["tool"]) for x in (1, 5, 25, 60, 300)},
       "idle_time_tool_triggered_in_windows_above": {str(x): sum(g for g in idle_gap["tool"] if g > x)/sum(idle_gap["tool"]) for x in (1, 5, 25, 60, 300)},
}
# --- tool latency by identity
rows = []
total_tool_time = sum(sum(v) for v in lat_tool.values())
for (p, name), xs in sorted(lat_tool.items(), key=lambda kv: -sum(kv[1])):
    rows.append({"provider": p, "tool": name, "calls": len(xs), "p50": pct(xs,.5), "p90": pct(xs,.9), "p99": pct(xs,.99),
                 "share_of_calls": len(xs)/calls_total, "share_of_tool_time": sum(xs)/total_tool_time,
                 "frac_long": sum(1 for x in xs if x > LONG_S)/len(xs)})
out["tool_latency_by_tool"] = rows[:25]
out["long_call_definition_s"] = LONG_S
# --- how well does tool identity predict a long wait?  (identity = tool name + first executable for shell tools)
all_calls = [(k, x) for k, xs in lat.items() for x in xs]
p_long = {k: sum(1 for x in xs if x > LONG_S)/len(xs) for k, xs in lat.items()}
med = {k: pct(xs,.5) for k, xs in lat.items()}
tp = fp = fn = tn = 0; long_time_caught = 0.0; long_time_total = 0.0; long_time_missed_pinned = 0.0
for k, x in all_calls:
    pred_long = p_long[k] >= 0.5
    is_long = x > LONG_S
    if is_long: long_time_total += x
    if pred_long and is_long: tp += 1; long_time_caught += x
    elif pred_long and not is_long: fp += 1
    elif not pred_long and is_long: fn += 1; long_time_missed_pinned += x
    else: tn += 1
base_long = sum(1 for _, x in all_calls if x > LONG_S)/len(all_calls)
out["long_wait_predictability_from_tool_identity"] = {
    "identity": "provider + tool_name + first executable (shell) ; predict long if group P(long)>=0.5",
    "base_rate_long": base_long, "accuracy": (tp+tn)/len(all_calls),
    "precision_long": tp/(tp+fp) if tp+fp else None, "recall_long": tp/(tp+fn) if tp+fn else None,
    "share_of_long_wait_time_correctly_predicted": long_time_caught/long_time_total if long_time_total else None,
    "distinct_identities": len(lat),
}
# top shell executables by total time
exe_rows = []
for (p, name, exe), xs in lat.items():
    if exe is None: continue
    exe_rows.append({"provider": p, "tool": name, "exe": exe, "calls": len(xs), "p50": pct(xs,.5), "p90": pct(xs,.9), "frac_long": p_long[(p,name,exe)], "share_of_tool_time": sum(xs)/total_tool_time})
exe_rows.sort(key=lambda r: -r["share_of_tool_time"])
out["shell_by_first_executable_top"] = exe_rows[:20]
json.dump(out, open("tracelab_decomposition.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("rounds","sessions","tool_calls_with_latency","steps_with_tools","step_generation_ms","step_tool_parallel_ms","tool_share_of_step_wallclock","by_provider","idle_window_s","idle_window_tool_triggered_share_above","idle_time_tool_triggered_in_windows_above","long_wait_predictability_from_tool_identity")}, indent=1))
