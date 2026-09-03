#!/usr/bin/env python3
"""Aggregate replay records into the experiment's headline numbers."""
import json, glob, sys, statistics as st
from replay import same_cmd, canon as canon_cmd
from collections import Counter, defaultdict
recs=[]
for f in sorted(glob.glob("runs/*.jsonl")):
    for line in open(f):
        r=json.loads(line)
        if "summary" in r: r["_file"]=f; recs.append(r)
def pct(xs,p):
    xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))] if xs else None
out={"sessions":len(recs),"by_framework":{}}
groups=defaultdict(list)
for r in recs: groups[r.get("framework") or "?"].append(r)
groups["all"]=recs
for g,rs in groups.items():
    s=Counter(); D=[]; eq=0; eqn=0; wasted=[]; edits=0; testish=[]; tool=[]; sessions_with_hit=0; hit_ds_by_cmd=Counter()
    saved=defaultdict(float); real_D=[]
    for r in rs:
        sm=r["summary"]; s["launches"]+=sm["launches"]; s["hits"]+=sm["hits"]; s["misses"]+=sm["misses"]; s["calls"]+=sm["calls"]
        tool.append(sm["total_tool_s"]); testish.append(sm["testish_s"])
        if sm["hits"]: sessions_with_hit+=1
        edits+=sum(1 for c in r["calls"] if c.get("changed_tree"))
        for e in r["spec_events"]:
            if e["kind"]=="hit":
                D.append(e["spec_duration_s"]); real_D.append(e["real_duration_s"]); eq+=e["output_equal_exact"]; eqn+=(e["output_equal_normalized"] or e.get("output_equal_lines_sorted", False))
                for k,v in e["saved_under_model_time"].items(): saved[k]+=v
            elif e["kind"]=="miss": wasted.append(e["wasted_s"])
    # post-edit opportunity analysis: after each tree-changing call, what is the next state-changing action?
    opp=Counter()
    for r in rs:
        calls=r["calls"]; last_test=None
        for i,c in enumerate(calls):
            if c.get("testish") and c.get("rc") is not None: last_test=c["cmd"]
            if c.get("changed_tree"):
                nxt=next((x for x in calls[i+1:] if not x["readonly"]), None)
                if nxt is None: opp["edit_then_end"]+=1
                elif nxt.get("changed_tree"): opp["edit_then_edit"]+=1
                elif (nxt.get("testish") or nxt.get("spec")=="hit") and last_test and same_cmd(nxt["cmd"], last_test): opp["edit_then_same_test"]+=1
                elif nxt.get("testish") or nxt.get("spec")=="hit": opp["edit_then_other_test"]+=1
                else: opp["edit_then_other"]+=1
    # policy variants from the same records: launch after any edit (measured) vs only after modifications of existing files
    bykind=defaultdict(Counter)
    for r in rs:
        for e in r["spec_events"]:
            if e["kind"] in ("hit","miss"): bykind[e.get("edit_kind") or "unknown"][e["kind"]]+=1
    policy={k: {"launches": v["hit"]+v["miss"], "hits": v["hit"], "hit_rate": v["hit"]/(v["hit"]+v["miss"]) if (v["hit"]+v["miss"]) else None} for k,v in bykind.items()}
    # Policy B (post hoc): after CREATING a file, predict that the agent runs it: `python <path>` or `pytest <path>`.
    import re as _re
    pb=Counter(); pb_D=[]
    def created_path(c):
        cmd=str(c.get("cmd") or "")
        if c["name"]=="str_replace_editor": return c.get("path") or None
        m=_re.search(r"cat\s*(?:<<\s*['\"]?\w+['\"]?)?\s*>\s*(\S+)", cmd) or _re.search(r">\s*(\S+\.py)\b", cmd) or _re.search(r"tee\s+(\S+)", cmd)
        return m.group(1).strip("'\"") if m else None
    for r in rs:
        calls=r["calls"]
        for i,c in enumerate(calls):
            if c.get("edit_kind")!="created": continue
            path=created_path(c)
            nxt=next((x for x in calls[i+1:] if not x["readonly"]), None)
            if not path or not nxt: pb["no_prediction"]+=1; continue
            pb["predicted"]+=1
            nc=str(nxt.get("cmd") or "")
            if nxt["name"]!="str_replace_editor" and _re.search(r"^(python[0-9.]*|pytest)\s+(-m\s+pytest\s+)?" + _re.escape(path.split("/")[-1]) + r"\b", canon_cmd(nc)):
                pb["hit"]+=1; pb_D.append(nxt.get("duration_s",0))
            else: pb["miss"]+=1
    n=len(rs)
    out["by_framework"][g]={"post_edit_next_action": dict(opp), "hit_rate_by_launching_edit_kind": policy,
        "policy_B_run_created_file": {"created_edits_with_prediction": pb["predicted"], "hits": pb["hit"], "misses": pb["miss"], "hit_rate": (pb["hit"]/pb["predicted"]) if pb["predicted"] else None, "no_prediction": pb["no_prediction"], "hit_D_p50": pct(pb_D,.5), "hit_D_p90": pct(pb_D,.9)}, "post_edit_same_test_share": (opp["edit_then_same_test"]/sum(opp.values())) if sum(opp.values()) else None,"sessions":n,"calls":s["calls"],"tree_changing_calls":edits,"launches":s["launches"],"hits":s["hits"],"misses":s["misses"],
        "hit_rate_per_launch": (s["hits"]/s["launches"]) if s["launches"] else None,
        "launches_per_edit": (s["launches"]/edits) if edits else None,
        "sessions_with_at_least_one_hit": sessions_with_hit/n if n else None,
        "hit_duration_s": {"n":len(D),"p50":pct(D,.5),"p90":pct(D,.9),"max":max(D) if D else None,"mean":st.mean(D) if D else None},
        "hit_output_equal_exact": (eq/len(D)) if D else None, "hit_output_equal_normalized": (eqn/len(D)) if D else None,
        "wasted_s_per_miss_mean": st.mean(wasted) if wasted else None,
        "tool_time_per_session_mean_s": st.mean(tool) if tool else None, "testish_time_per_session_mean_s": st.mean(testish) if testish else None,
        "saved_per_session_under_model_time_s": {k: v/n for k,v in saved.items()},
        "saved_share_of_tool_time_under_model_time": {k: (v/sum(tool)) if sum(tool) else None for k,v in saved.items()}}
json.dump(out, open("runs/summary.json","w"), indent=1)
print(json.dumps(out, indent=1))
