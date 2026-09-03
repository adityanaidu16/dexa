#!/usr/bin/env python3
"""Live coding-agent harness with speculative post-edit test execution, on SWE-bench Verified task images.

Tools: bash (in the task container) and str_replace_editor (view/create/str_replace/insert). The model is
called through the Anthropic SDK. With --spec on, after a successful edit the harness immediately launches
the most recent test-like command in the background; if the model's next state-changing action is that
same command, its output is served from the speculative run (a hit). Per task we record wall-clock, model
time, tool time, tokens, cost, hits/misses and realized savings, then grade FAIL_TO_PASS.
"""
import json, re, sys, time, subprocess, shlex, os, argparse
import anthropic
from replay import Container, is_testish, is_readonly, norm

PRICES = {"claude-opus-5": (5.0, 25.0, 0.5, 6.25), "claude-sonnet-5": (2.0, 10.0, 0.2, 2.5), "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25)}  # $/M: input, output, cache read, cache write

TOOLS = [
    {"name": "bash", "description": "Run a bash command in the repository container (cwd /testbed). Long-running commands time out after 600 s.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}, "strict": True},
    {"name": "str_replace_editor", "description": "View, create, and edit files. commands: view (path, optional view_range [start,end]), create (path, file_text), str_replace (path, old_str, new_str; old_str must match exactly once), insert (path, insert_line, new_str).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]}, "path": {"type": "string"},
                      "file_text": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}, "insert_line": {"type": "integer"},
                      "view_range": {"type": "array", "items": {"type": "integer"}}}, "required": ["command", "path"], "additionalProperties": False}, "strict": True},
    {"name": "submit", "description": "Call when the fix is complete and tests you wrote pass.", "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}, "strict": True},
]
SYSTEM = """You are an expert software engineer fixing a GitHub issue in a Python repository checked out at /testbed inside a container.
Work method: explore the relevant code, write a small script that reproduces the issue, make the minimal fix, re-run your reproduction script and the relevant existing tests, then call submit.
Use the bash tool for commands and str_replace_editor for viewing and editing files. Keep tool calls purposeful; do not narrate at length."""

def usage_cost(u, model):
    p = PRICES[model]
    return (u.input_tokens * p[0] + u.output_tokens * p[1] + (u.cache_read_input_tokens or 0) * p[2] + (u.cache_creation_input_tokens or 0) * p[3]) / 1e6

def run_task(inst, image, model, spec, cname, max_steps=40, effort="medium", per_cmd_timeout=600):
    client = anthropic.Anthropic()
    c = Container(image, cname)
    rec = {"instance_id": inst["instance_id"], "model": model, "spec": spec, "steps": [], "spec_events": [], "cost_usd": 0.0, "tokens": {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}}
    t_start = time.time(); model_s = 0.0; tool_s = 0.0
    try:
        messages = [{"role": "user", "content": f"<issue>\n{inst['problem_statement']}\n</issue>\nFix this issue in the repository at /testbed."}]
        fp = c.tree_fingerprint(); last_test = None; pending = None; submitted = False
        for step in range(max_steps):
            t0 = time.time()
            with client.messages.stream(model=model, max_tokens=16000, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                        tools=TOOLS, messages=messages, output_config={"effort": effort}) as stream:
                resp = stream.get_final_message()
            dt_model = time.time() - t0; model_s += dt_model
            u = resp.usage; rec["cost_usd"] += usage_cost(u, model)
            rec["tokens"]["in"] += u.input_tokens; rec["tokens"]["out"] += u.output_tokens
            rec["tokens"]["cache_read"] += (u.cache_read_input_tokens or 0); rec["tokens"]["cache_write"] += (u.cache_creation_input_tokens or 0)
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                rec["steps"].append({"step": step, "model_s": dt_model, "end": resp.stop_reason}); break
            results = []
            for tu in tool_uses:
                name, a = tu.name, tu.input
                cmd = (a.get("command") or "").strip() if name == "bash" else None
                entry = {"step": step, "model_s": dt_model, "tool": name, "cmd": cmd or a.get("command"), "readonly": is_readonly("str_replace_editor" if name == "str_replace_editor" else "bash", a)}
                if name == "submit":
                    submitted = True; results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "submitted"}); rec["steps"].append(entry); break
                # resolve a pending speculative run
                served = None
                if pending and not entry["readonly"]:
                    if name == "bash" and cmd == pending["cmd"]:
                        t_req = time.time()
                        while not c.spec_poll_done():
                            if time.time() - pending["t0"] > per_cmd_timeout: c.spec_kill(); break
                            time.sleep(0.3)
                        t_done = time.time(); D = t_done - pending["t0"]; wait = max(0.0, t_done - t_req)
                        spec_rc, spec_out = c.spec_result(); served = (int(spec_rc) if spec_rc.strip().isdigit() else 1, spec_out, wait)
                        rec["spec_events"].append({"step": step, "kind": "hit", "cmd": cmd, "spec_duration_s": D, "waited_s": wait, "saved_s": max(0.0, D - wait)})
                        pending = None
                    else:
                        c.spec_kill(); rec["spec_events"].append({"step": step, "kind": "miss", "cmd": pending["cmd"], "next": entry["cmd"], "wasted_s": time.time() - pending["t0"]}); pending = None
                if served:
                    rc, outp, dt = served[0], served[1], served[2]
                elif name == "bash":
                    rc, outp, dt = c.exec(cmd, timeout=per_cmd_timeout)
                else:
                    rc, outp, dt = c.editor(a)
                tool_s += dt
                outp = outp if len(outp) <= 30000 else outp[:15000] + "\n...[truncated]...\n" + outp[-15000:]
                entry.update({"rc": rc, "tool_s": dt, "out_chars": len(outp), "served_from_spec": bool(served)})
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": outp or "(no output)", "is_error": rc not in (0, None)})
                if cmd and is_testish(cmd): last_test = cmd
                if not entry["readonly"]:
                    nfp = c.tree_fingerprint(); changed = nfp != fp; fp = nfp; entry["changed_tree"] = changed
                    if spec and changed and last_test and cmd != last_test and rc in (0, None):
                        pending = {"cmd": last_test, "t0": c.spec_start(last_test)}; rec["spec_events"].append({"step": step, "kind": "launch", "cmd": last_test})
                rec["steps"].append(entry)
            messages.append({"role": "user", "content": results})
            if submitted: break
        if pending: c.spec_kill()
        # grade: apply the test patch and run FAIL_TO_PASS
        rc, diff, _ = c.exec("git diff", timeout=60); rec["model_patch"] = diff
        rec["grade"] = grade(c, inst)
    finally:
        c.close()
    rec.update({"wall_s": time.time() - t_start, "model_s": model_s, "tool_s": tool_s, "submitted": submitted,
                "hits": sum(e["kind"] == "hit" for e in rec["spec_events"]), "misses": sum(e["kind"] == "miss" for e in rec["spec_events"]),
                "saved_s": sum(e.get("saved_s", 0) for e in rec["spec_events"])})
    return rec

def grade(c, inst):
    """Apply the gold test patch and run the FAIL_TO_PASS tests with the repo's test command."""
    subprocess.run(["docker", "exec", "-i", "-w", "/testbed", c.name, "bash", "-lc", "cat > /tmp/test.patch"], input=inst["test_patch"], text=True)
    rc, o, _ = c.exec("git apply /tmp/test.patch", timeout=60)
    tests = json.loads(inst["FAIL_TO_PASS"]) if isinstance(inst["FAIL_TO_PASS"], str) else inst["FAIL_TO_PASS"]
    cmd = inst.get("test_cmd") or "pytest -rA"
    rc, out, dt = c.exec(f"{cmd} {' '.join(shlex.quote(t) for t in tests[:20])}", timeout=900)
    passed = sum(1 for t in tests[:20] if re.search(r"PASSED\s+" + re.escape(t.split('::')[-1]) + r"|" + re.escape(t) + r"\s+PASSED|OK", out))
    return {"rc": rc, "fail_to_pass": len(tests), "passed_heuristic": passed, "duration_s": dt, "tail": out[-1500:]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True); ap.add_argument("--ids", required=True, help="comma-separated instance ids")
    ap.add_argument("--model", default="claude-opus-5"); ap.add_argument("--effort", default="medium"); ap.add_argument("--spec", choices=["on", "off"], default="on")
    ap.add_argument("--out", required=True); ap.add_argument("--worker", default="live0"); ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()
    inst_by_id = {r["instance_id"]: r for r in json.load(open(args.instances))}
    for iid in args.ids.split(","):
        inst = inst_by_id[iid]
        image = f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"
        subprocess.run(["docker", "pull", image], capture_output=True)
        rec = run_task(inst, image, args.model, args.spec == "on", f"live_{args.worker}", max_steps=args.max_steps, effort=args.effort)
        open(args.out, "a").write(json.dumps(rec) + "\n")
        print(f"{iid} spec={args.spec} wall={rec['wall_s']:.0f}s model={rec['model_s']:.0f}s tool={rec['tool_s']:.0f}s hits={rec['hits']} misses={rec['misses']} saved={rec['saved_s']:.0f}s cost=${rec['cost_usd']:.2f} grade={rec.get('grade',{}).get('passed_heuristic')}/{rec.get('grade',{}).get('fail_to_pass')}", flush=True)
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
