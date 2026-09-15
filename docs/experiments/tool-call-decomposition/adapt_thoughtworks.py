#!/usr/bin/env python3
"""Convert SWE-agent / mini-SWE-agent text-formatted trajectories into the OpenHands-style
(assistant.tool_calls + tool message) structure so the same analyzers run on them."""
import json, re, sys, shlex
rows=json.load(open("thoughtworks_600.json"))
FENCE=re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n```", re.S)
out=[]
for r in rows:
    m=json.loads(r["messages_json"]) if isinstance(r["messages_json"],str) else r["messages_json"]
    traj=[]; k=0
    for i,x in enumerate(m):
        role=x.get("role")
        if role in ("system","user") and not (i>0 and m[i-1].get("role")=="assistant" and traj and traj[-1].get("role")=="assistant" and traj[-1].get("tool_calls")):
            traj.append({"role":role,"content":x.get("content") or ""}); continue
        if role=="assistant":
            content=x.get("content") or ""
            f=FENCE.search(content)
            if not f:
                traj.append({"role":"assistant","content":content,"tool_calls":[]}); continue
            cmd=f.group(1).strip()
            name="execute_bash"; args={"command":cmd}
            if cmd.startswith("str_replace_editor"):
                try: parts=shlex.split(cmd)
                except Exception: parts=cmd.split()
                sub=parts[1] if len(parts)>1 else None; path=parts[2] if len(parts)>2 else None
                args={"command":sub,"path":path}
                if sub=="str_replace":
                    # --old_str ... --new_str ...
                    mo=re.search(r"--old_str\s+(.*?)\s+--new_str\s+(.*)$", cmd, re.S)
                    if mo: args["old_str"]=mo.group(1); args["new_str"]=mo.group(2)
                elif sub=="view":
                    mv=re.search(r"--view_range\s+\[?([0-9]+),?\s*([0-9]+)", cmd)
                    if mv: args["view_range"]=[int(mv.group(1)),int(mv.group(2))]
                elif sub=="create":
                    mc=re.search(r"--file_text\s+(.*)$", cmd, re.S)
                    if mc: args["file_text"]=mc.group(1)
                name="str_replace_editor"
            k+=1; cid=f"c{k}"
            traj.append({"role":"assistant","content":content[:f.start()],"tool_calls":[{"id":cid,"function":{"name":name,"arguments":json.dumps(args)}}]})
            # the observation is the next user message
            if i+1 < len(m) and m[i+1].get("role")=="user":
                obs=m[i+1].get("content") or ""
                obs=re.sub(r"^(OBSERVATION:\s*|<returncode>\d+</returncode>\s*<output>\s*)", "", obs)
                traj.append({"role":"tool","tool_call_id":cid,"name":name,"content":obs})
    out.append({"trajectory_id":r["session_id"],"framework":r["agent_framework"],"model":r["recorded_model"],"trajectory":traj})
json.dump(out,open("thoughtworks_adapted.json","w"))
from collections import Counter
print("adapted", len(out), Counter((o["framework"],o["model"]) for o in out))
print("sample calls:", sum(1 for o in out for x in o["trajectory"] if x.get("role")=="tool"))
