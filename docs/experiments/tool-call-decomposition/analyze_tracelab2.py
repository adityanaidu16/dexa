#!/usr/bin/env python3
"""TraceLab v2: separate machine tool time from human waits and subagent waits; refined tool identity."""
import gzip, json, sys
from datetime import datetime
from collections import defaultdict, Counter
def ts(s): return datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()
def pct(xs,p):
    if not xs: return None
    xs=sorted(xs); k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1); return xs[f]+(xs[c]-xs[f])*(k-f)
HUMAN={"AskUserQuestion","ExitPlanMode","request_user_input","ask_user"}
SUBAGENT={"Agent","TaskOutput","wait_agent","Task","spawn_agent","send_input","TaskStop","close_agent"}
SHELL={"Bash","exec_command","write_stdin","shell","BashOutput","KillShell"}
FILE={"Read","Edit","Write","MultiEdit","Glob","Grep","apply_patch","NotebookEdit","LS","view_image","read_file"}
WEB={"WebFetch","WebSearch","web_search","fetch"}
SKIP_EXE={"cd","echo","export","source","set","true","cat","printf","pwd","ls","timeout","env","time","nohup","sudo"}
TESTISH={"pytest","py.test","npm","npx","cargo","go","make","jest","mvn","gradle","tox","docker","python-script","python","python3","node","uv","poetry","pnpm","yarn","ruff","mypy","tsc","bun","dotnet","php","ruby","rspec","swift","ctest","cmake","bazel","flutter","conda","pip","pip3","git"}
def cat_of(name):
    if name in HUMAN: return "human"
    if name in SUBAGENT: return "subagent"
    if name in SHELL: return "shell"
    if name in FILE: return "file"
    if name in WEB: return "web"
    return "other"
def ident(name, exes):
    if name in SHELL and exes:
        for e in exes:
            if e not in SKIP_EXE: return f"{name}:{e}"
        return f"{name}:{exes[0]}"
    return name
LONG=5.0
sess=defaultdict(list); n=0
with gzip.open("syfi_coding_trace.jsonl.gz","rt") as f:
    for line in f:
        r=json.loads(line); n+=1
        ev=r.get("timing_events") or []
        if not ev: continue
        t0=ts(ev[0]["timestamp"]); asst=[ts(e["timestamp"]) for e in ev if e["event_type"] in ("reasoning","text","tool_call")]
        sess[r["session_id"]].append({"p":r["provider"],"i":r["round_index"],"t0":t0,"ta":max(asst) if asst else None,"fi":r.get("first_input_event_type"),
            "tools":[(t.get("tool_name"),t.get("tool_wall_latency_ms"),t.get("executables") or []) for t in (r.get("tools") or [])]})
tot=defaultdict(float); steps=Counter(); share=[]; long_share_steps=Counter()
lat=defaultdict(list); latcat=defaultdict(list); idle={"tool_machine":[], "tool_any":[], "user":[]}
test_time=0.0; shell_time=0.0
for sid, rs in sess.items():
    rs.sort(key=lambda r:r["i"])
    for i,r in enumerate(rs):
        if r["ta"] is None: continue
        g=(r["ta"]-r["t0"])
        if g<0 or g>3600: continue
        machine=[]; human_or_sub=[]
        for (name,w,exes) in r["tools"]:
            if w is None or w<0: continue
            w=w/1000.0; c=cat_of(name)
            lat[(r["p"],ident(name,exes))].append(w); latcat[(r["p"],c)].append(w)
            if c in ("human","subagent"): human_or_sub.append(w)
            else:
                machine.append(w)
                if c=="shell":
                    shell_time+=w
                    if any(e in TESTISH for e in exes): test_time+=w
        if machine:
            tm=max(machine)
            if g+tm>0:
                steps[r["p"]]+=1; tot[(r["p"],"gen")]+=g; tot[(r["p"],"tool")]+=tm; share.append(tm/(g+tm))
                if tm>g: long_share_steps[r["p"]]+=1
        if i+1<len(rs):
            gap=rs[i+1]["t0"]-r["ta"]
            if 0<=gap<=24*3600:
                if rs[i+1]["fi"]=="tool_result":
                    idle["tool_any"].append(gap)
                    if not human_or_sub: idle["tool_machine"].append(gap)
                else: idle["user"].append(gap)
def summ(xs): return {"n":len(xs),"p50":pct(xs,.5),"p90":pct(xs,.9),"p99":pct(xs,.99),"mean":sum(xs)/len(xs) if xs else None}
out={"rounds":n,"sessions":len(sess),
 "machine_tool_share_time_weighted":{p:tot[(p,"tool")]/(tot[(p,"gen")]+tot[(p,"tool")]) for p in steps},
 "machine_tool_share_per_step_median":pct(share,.5),"steps_where_machine_tool_exceeds_generation":{p:long_share_steps[p]/steps[p] for p in steps},
 "mean_seconds_per_step":{p:{"generation":tot[(p,"gen")]/steps[p],"machine_tool":tot[(p,"tool")]/steps[p]} for p in steps},
 "tool_time_by_category":{},
 "idle_window_s":{k:summ(v) for k,v in idle.items()},
 "idle_machine_share_above":{str(x):sum(1 for g in idle["tool_machine"] if g>x)/len(idle["tool_machine"]) for x in (1,5,25,60)},
 "idle_machine_time_in_windows_above":{str(x):sum(g for g in idle["tool_machine"] if g>x)/sum(idle["tool_machine"]) for x in (1,5,25,60)},
 "testish_share_of_shell_time":test_time/shell_time if shell_time else None}
total_all=sum(sum(v) for v in latcat.values())
for (p,c),xs in sorted(latcat.items(), key=lambda kv:-sum(kv[1])):
    out["tool_time_by_category"][f"{p}:{c}"]={"calls":len(xs),"p50":pct(xs,.5),"p90":pct(xs,.9),"p99":pct(xs,.99),"share_of_all_tool_time":sum(xs)/total_all,"frac_long":sum(1 for x in xs if x>LONG)/len(xs)}
# predictability of long waits from refined identity, machine tools only
calls=[(k,x) for k,xs in lat.items() for x in xs if cat_of(k[1].split(":")[0]) not in ("human","subagent")]
pl={k:sum(1 for x in xs if x>LONG)/len(xs) for k,xs in lat.items()}
tp=fp=fn=tn=0; caught=0.0; ltot=0.0
for k,x in calls:
    pred=pl[k]>=0.5; isl=x>LONG
    if isl: ltot+=x
    if pred and isl: tp+=1; caught+=x
    elif pred: fp+=1
    elif isl: fn+=1
    else: tn+=1
out["long_wait_predictability_machine_tools"]={"identity":"tool + first non-trivial executable","base_rate_long":sum(1 for _,x in calls if x>LONG)/len(calls),
  "accuracy":(tp+tn)/len(calls),"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,"share_of_long_wait_time_predicted":caught/ltot,"identities":len(set(k for k,_ in calls))}
rows=[]
for k,xs in lat.items():
    if cat_of(k[1].split(":")[0]) in ("human","subagent"): continue
    rows.append({"id":f"{k[0]}:{k[1]}","calls":len(xs),"p50":pct(xs,.5),"p90":pct(xs,.9),"frac_long":pl[k],"share_of_machine_tool_time":sum(xs)/sum(x for _,x in calls)})
rows.sort(key=lambda r:-r["share_of_machine_tool_time"]); out["machine_tool_identities_top"]=rows[:20]
json.dump(out,open("tracelab_decomposition_v2.json","w"),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!="machine_tool_identities_top"},indent=1))
print("TOP MACHINE IDENTITIES")
for r in rows[:16]: print(f'  {r["id"]:34} calls={r["calls"]:7} p50={r["p50"]:.2f}s p90={r["p90"]:.1f}s long%={100*r["frac_long"]:.0f} time%={100*r["share_of_machine_tool_time"]:.1f}')
