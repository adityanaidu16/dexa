#!/usr/bin/env python3
"""Structural analysis of coding-agent trajectories with tool contents (no timing):
for each tool call: is the result reconstructible from prior context, does the next action use new
information from the result, is the next action an exact repeat, is the tool read-only."""
import json, re, sys
from collections import Counter, defaultdict

SRC = sys.argv[1]; OUT = sys.argv[2]
rows = json.load(open(SRC))

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{3,}")
READONLY_EXE = {"cat","ls","grep","rg","find","head","tail","wc","pwd","which","tree","stat","file","diff","echo","printf",
                "git","python","python3","pytest","py.test","pip","sed","awk","cd","export","source","true","test","ag","less","more","md5sum","sha256sum","nl","cut","sort","uniq","xargs","env","type"}
WRITE_HINTS = re.compile(r"(>>?\s*[^&|\s]|\bsed\s+-i|\btee\b|\bmv\b|\bcp\b|\brm\b|\bmkdir\b|\btouch\b|\bgit\s+(checkout|reset|stash|apply|commit|add|rm|mv|clean|revert|cherry-pick|merge|rebase|pull)|\bpip\s+install|\bpython[0-9.]*\s+(?!-c\s+['\"]print)|\bpytest\b|\bmake\b|\bnpm\b|\bcargo\b|\bpatch\b|\bchmod\b|\bln\b|\bapply_patch\b)")

def is_readonly_bash(cmd):
    if not cmd: return False
    if WRITE_HINTS.search(cmd): return False
    return True

def norm_call(name, args):
    try: a = json.loads(args) if isinstance(args, str) else (args or {})
    except Exception: a = {"_raw": args}
    return name, a

def call_key(name, a):
    return json.dumps([name, a], sort_keys=True, default=str)

agg = Counter(); by_tool = defaultdict(Counter); dup_check = Counter()
n_traj = 0
for r in rows:
    msgs = r.get("trajectory") or r.get("messages") or []
    if not msgs: continue
    n_traj += 1
    # linearize into (assistant call, tool result) pairs in order
    calls = []   # dicts: name, args(dict), argstr, result, ctx_before (index into text list)
    context_texts = []   # all text seen so far (system/user/assistant content, tool results, call args)
    pending = {}
    for m in msgs:
        role = m.get("role")
        if role in ("system", "user"):
            context_texts.append(m.get("content") or "")
        elif role == "assistant":
            context_texts.append(m.get("content") or "")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                name, a = norm_call(fn.get("name"), fn.get("arguments"))
                argstr = json.dumps(a, default=str)
                pending[tc.get("id") or tc.get("tool_call_id")] = {"name": name, "args": a, "argstr": argstr, "ctx_len": len(context_texts)}
                context_texts.append(argstr)
        elif role == "tool":
            cid = m.get("tool_call_id")
            c = pending.pop(cid, None)
            if c is None: continue
            c["result"] = m.get("content") or ""
            c["result_ctx_index"] = len(context_texts)
            context_texts.append(c["result"])
            calls.append(c)
    # per-trajectory state for reconstructibility
    file_state_known = {}     # path -> last known content signature (from view/create/edit results) valid until modified
    seen_results = set()
    seen_calls = set()
    last_cmd_result = {}      # command -> (result, epoch)
    epoch = 0                 # increments on any potentially modifying action
    for i, c in enumerate(calls):
        name = c["name"]; a = c["args"]; res = c["result"]
        sub = a.get("command") if name in ("str_replace_editor", "str_replace_based_edit_tool", "text_editor") else None
        tool_label = f"{name}:{sub}" if sub in ("view","create","str_replace","insert","undo_edit") else name
        agg["calls"] += 1; by_tool[tool_label]["calls"] += 1
        # read-only?
        if name in ("execute_bash", "bash", "shell", "run"):
            cmd = a.get("command") or a.get("cmd") or ""
            ro = is_readonly_bash(cmd)
        elif tool_label.endswith(":view") or name in ("think", "finish", "browser"):
            ro = True
        else:
            ro = False
        if ro: agg["readonly"] += 1; by_tool[tool_label]["readonly"] += 1
        # exact duplicate result seen before in this trajectory
        rsig = (res or "").strip()
        if rsig and rsig in seen_results:
            agg["dup_result"] += 1; by_tool[tool_label]["dup_result"] += 1
        # reconstructible from prior context
        recon = False
        if tool_label.endswith(":view"):
            path = a.get("path"); 
            if path in file_state_known and file_state_known[path] == epoch:
                recon = True
        elif tool_label.endswith((":create", ":str_replace", ":insert", ":undo_edit")):
            # editor echo is deterministic when the edit succeeds
            recon = not re.search(r"(No replacement|did not appear|Invalid|Error|not found|Cannot|already exists)", res[:300], re.I)
        elif name in ("think", "finish"):
            recon = True
        elif name in ("execute_bash", "bash", "shell", "run"):
            cmd = (a.get("command") or "").strip()
            if cmd in last_cmd_result and last_cmd_result[cmd][1] == epoch:
                recon = True
                dup_check["repeat_cmd"] += 1
                if last_cmd_result[cmd][0].strip() == rsig: dup_check["repeat_cmd_same_output"] += 1
        if recon: agg["reconstructible"] += 1; by_tool[tool_label]["reconstructible"] += 1
        # update state
        if tool_label.endswith(":view"):
            file_state_known[a.get("path")] = epoch
        elif tool_label.endswith((":create", ":str_replace", ":insert", ":undo_edit")):
            epoch += 1; file_state_known[a.get("path")] = epoch
        elif name in ("execute_bash", "bash", "shell", "run"):
            cmd = (a.get("command") or "").strip()
            if not ro: epoch += 1
            last_cmd_result[cmd] = (res, epoch)
        seen_results.add(rsig)
        # next action: does it use new info from this result?
        if i + 1 < len(calls):
            nxt = calls[i+1]
            # context strictly before this result
            before = "\n".join(context_texts[:c["result_ctx_index"]])
            new_in_result = set(t for t in TOKEN.findall(res) if t not in before)
            nxt_tokens = set(TOKEN.findall(nxt["argstr"]))
            uses_new = bool(nxt_tokens & new_in_result)
            agg["has_next"] += 1; by_tool[tool_label]["has_next"] += 1
            if uses_new: agg["next_uses_new_info"] += 1; by_tool[tool_label]["next_uses_new_info"] += 1
            nk = call_key(nxt["name"], nxt["args"])
            if nk in seen_calls: agg["next_is_exact_repeat"] += 1; by_tool[tool_label]["next_is_exact_repeat"] += 1
            if nxt["name"] in ("think","finish"): agg["next_is_think_or_finish"] += 1
        seen_calls.add(call_key(name, a))

def frac(c, k, d="calls"): return (c[k]/c[d]) if c.get(d) else None
out = {"trajectories": n_traj, "calls": agg["calls"],
       "readonly_fraction": frac(agg,"readonly"),
       "result_reconstructible_fraction": frac(agg,"reconstructible"),
       "result_exact_duplicate_fraction": frac(agg,"dup_result"),
       "next_action_uses_new_info_from_result": frac(agg,"next_uses_new_info","has_next"),
       "next_action_independent_of_result": 1 - frac(agg,"next_uses_new_info","has_next"),
       "next_action_exact_repeat_of_earlier": frac(agg,"next_is_exact_repeat","has_next"),
       "next_action_is_think_or_finish": frac(agg,"next_is_think_or_finish","has_next"),
       "repeated_command_same_output": (dup_check["repeat_cmd_same_output"]/dup_check["repeat_cmd"]) if dup_check["repeat_cmd"] else None,
       "repeated_commands": dup_check["repeat_cmd"],
       "by_tool": {t: {"calls": c["calls"], "share": c["calls"]/agg["calls"], "readonly": frac(c,"readonly"), "reconstructible": frac(c,"reconstructible"),
                       "dup_result": frac(c,"dup_result"), "next_uses_new_info": frac(c,"next_uses_new_info","has_next"), "next_exact_repeat": frac(c,"next_is_exact_repeat","has_next")}
                   for t, c in sorted(by_tool.items(), key=lambda kv: -kv[1]["calls"])}}
json.dump(out, open(OUT,"w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
