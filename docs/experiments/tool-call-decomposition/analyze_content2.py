#!/usr/bin/env python3
"""Control-dependence proxies on OpenHands trajectories: does the next action's TYPE depend on whether a shell
result failed? What is the next action after an edit? What do exact-repeat next actions look like?"""
import json, re, sys
from collections import Counter, defaultdict
rows=json.load(open(sys.argv[1]))
FAIL=re.compile(r"(Traceback|Error|FAILED|failed|error:|Exception|not found|No such file|exit code [1-9]|AssertionError|\bFAIL\b)", re.I)
def parse(msgs):
    pend={}; calls=[]
    for m in msgs:
        if m.get("role")=="assistant":
            for tc in (m.get("tool_calls") or []):
                fn=tc.get("function") or {}
                try: a=json.loads(fn.get("arguments"))
                except Exception: a={"_raw":fn.get("arguments")}
                pend[tc.get("id")]={"name":fn.get("name"),"args":a}
        elif m.get("role")=="tool":
            c=pend.pop(m.get("tool_call_id"),None)
            if c: c["result"]=m.get("content") or ""; calls.append(c)
    return calls
def kind(c):
    n=c["name"]; a=c["args"]
    if n=="str_replace_editor": return {"view":"view","create":"edit","str_replace":"edit","insert":"edit","undo_edit":"edit"}.get(a.get("command"),"editor")
    if n=="execute_bash": return "shell"
    return n
cond=defaultdict(Counter); after_edit=Counter(); after_edit_repeat_kind=Counter(); repeat_cmds=Counter(); n_edit=0
shell_after_edit_is_repeat=0; shell_after_edit=0
for r in rows:
    calls=parse(r["trajectory"]); seen={}
    for i in range(len(calls)-1):
        c,nx=calls[i],calls[i+1]; k=kind(c); nk=kind(nx)
        key=json.dumps([nx["name"],nx["args"]],sort_keys=True,default=str)
        if k=="shell":
            failed=bool(FAIL.search(c["result"][-2000:]))
            same_cmd = (nk=="shell" and (nx["args"].get("command","").strip()==c["args"].get("command","").strip()))
            cond["failed" if failed else "ok"][("shell-same" if same_cmd else nk)]+=1
        if k=="edit":
            n_edit+=1; after_edit[nk]+=1
            if nk=="shell":
                shell_after_edit+=1
                if key in seen:
                    shell_after_edit_is_repeat+=1
                    cmd=nx["args"].get("command","")
                    ex=(cmd.strip().split() or ["?"])[0]
                    repeat_cmds[ex if ex not in ("cd","&&") else cmd.strip()[:40]]+=1
        seen[json.dumps([c["name"],c["args"]],sort_keys=True,default=str)]=i
    # count current too
        seen[key]=seen.get(key,i+1)
out={}
for cnd in ("ok","failed"):
    tot=sum(cond[cnd].values()); out[f"next_after_shell_{cnd}"]={k:v/tot for k,v in cond[cnd].most_common(8)}; out[f"n_shell_{cnd}"]=tot
tot=sum(after_edit.values()); out["next_after_edit"]={k:v/tot for k,v in after_edit.most_common(8)}
out["shell_after_edit_fraction"]=shell_after_edit/n_edit; out["shell_after_edit_is_exact_repeat"]=shell_after_edit_is_repeat/shell_after_edit if shell_after_edit else None
out["repeat_command_first_word_top"]=dict(repeat_cmds.most_common(12))
print(json.dumps(out,indent=1)); json.dump(out,open(sys.argv[2],"w"),indent=1)
