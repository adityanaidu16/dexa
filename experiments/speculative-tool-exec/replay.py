#!/usr/bin/env python3
"""Replay recorded coding-agent trajectories inside their real Docker task images with a speculative
post-edit test execution policy, and measure what it would have saved.

Policy: after any tool call that modifies the working tree, if the most recent test-like shell command
is known and safe, launch it immediately in the background. When the trajectory's next state-changing
action arrives: if it is that same command, count a HIT (its output is already computed); anything else
kills the speculative run and counts a MISS. Read-only commands in between do not disturb a pending run.

For every hit we also run the command for real afterwards and compare outputs (validation), and we record
the speculative run's duration D. The time a live agent would save is min(D, model time of the next step).
"""
import json, re, sys, time, subprocess, shlex, os, hashlib, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TESTISH = re.compile(r"(^|&&\s*|;\s*)(python[0-9.]*\s+(-m\s+pytest|-m\s+unittest|\S+\.py)|pytest|py\.test|tox\b|make\s+test|npm\s+test|cargo\s+test|go\s+test)")
UNSAFE = re.compile(r"(\brm\b|\bgit\s+(checkout|reset|stash|apply|commit|add|clean|revert)|pip\s+install|sed\s+-i|\btee\b|\bmv\b|\bcp\b|>\s*(?!&)[^&|\s]|\bmkdir\b|\btouch\b|\bchmod\b|\bpatch\b)")
READONLY_BASH = re.compile(r"^\s*(cd\s+\S+\s*&&\s*)?(cat|ls|grep|rg|find|head|tail|wc|pwd|which|tree|stat|git\s+(status|diff|log|show|branch)|sed\s+-n|nl|echo\s+[^>]*|python[0-9.]*\s+-c\s+['\"]print)")
NORMALIZE = [(re.compile(r"\d+\.\d+s\b"), "Xs"), (re.compile(r"0x[0-9a-fA-F]+"), "0xX"), (re.compile(r"/tmp/[\w./-]+"), "/tmp/X"),
             (re.compile(r"\b\d{2}:\d{2}:\d{2}\b"), "HH:MM:SS"), (re.compile(r"(passed|failed|error|skipped|warning)s? in [\d.]+s"), r"\1 in Xs"), (re.compile(r"\s+$", re.M), "")]

def norm(s):
    for pat, rep in NORMALIZE: s = pat.sub(rep, s)
    return s.strip()

class Container:
    def __init__(self, image, name, workdir="/testbed"):
        self.image = image; self.name = name; self.wd = workdir
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "run", "-d", "--name", name, image, "sleep", "infinity"], check=True, capture_output=True)
        subprocess.run(["docker", "cp", os.path.join(HERE, "editor.py"), f"{name}:/tmp/editor.py"], check=True, capture_output=True)
        subprocess.run(["docker", "cp", os.path.join(HERE, "edit_via_str_replace"), f"{name}:{workdir}/edit_via_str_replace"], check=True, capture_output=True)
        subprocess.run(["docker", "exec", "-w", workdir, name, "bash", "-lc", "chmod +x edit_via_str_replace; echo edit_via_str_replace >> .git/info/exclude"], capture_output=True)
    def exec(self, cmd, timeout=900, stdin=None):
        t0 = time.time()
        try:
            p = subprocess.run(["docker", "exec", "-i", "-w", self.wd, self.name, "bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout, input=stdin)
            rc, outp = p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            rc, outp = 124, ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n[timeout]"
        return rc, outp, time.time() - t0
    def apply_patch(self, patch):
        subprocess.run(["docker", "exec", "-i", "-w", self.wd, self.name, "bash", "-lc", "cat > /tmp/bug.patch"], input=patch, text=True, check=True)
        rc, o, _ = self.exec("git apply /tmp/bug.patch && git status --short | head -5")
        return rc, o
    def tree_fingerprint(self):
        rc, o, _ = self.exec("git status --porcelain=v1 2>/dev/null | md5sum; git diff 2>/dev/null | md5sum", timeout=60)
        return o.strip()
    def tree_status(self):
        rc, o, _ = self.exec("git status --porcelain=v1 2>/dev/null | head -200", timeout=60)
        return set(l for l in o.splitlines() if l.strip())
    def spec_start(self, cmd):
        wrapped = f"rm -f /tmp/spec.rc /tmp/spec.out; setsid bash -c {shlex.quote(cmd + '; echo $? > /tmp/spec.rc')} > /tmp/spec.out 2>&1 < /dev/null & echo $! > /tmp/spec.pid"
        rc, o, _ = self.exec(wrapped, timeout=30)
        return time.time()
    def spec_poll_done(self):
        rc, o, _ = self.exec("test -f /tmp/spec.rc && echo done || echo running", timeout=30)
        return "done" in o
    def spec_kill(self):
        self.exec("if [ -f /tmp/spec.pid ]; then kill -TERM -- -$(cat /tmp/spec.pid) 2>/dev/null; sleep 0.2; kill -KILL -- -$(cat /tmp/spec.pid) 2>/dev/null; fi; rm -f /tmp/spec.pid", timeout=30)
    def spec_result(self):
        rc, o, _ = self.exec("cat /tmp/spec.rc; echo ---SPECOUT---; cat /tmp/spec.out", timeout=60)
        head, _, body = o.partition("---SPECOUT---\n")
        return head.strip(), body
    def editor(self, args):
        rc, o, dt = self.exec("python3 /tmp/editor.py", timeout=120, stdin=json.dumps(args))
        return rc, o, dt
    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

def parse_calls(traj):
    pend = {}; calls = []
    for m in traj:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                try: a = json.loads(fn.get("arguments"))
                except Exception: a = {"_raw": fn.get("arguments")}
                pend[tc.get("id")] = {"name": fn.get("name"), "args": a}
        elif m.get("role") == "tool":
            c = pend.pop(m.get("tool_call_id"), None)
            if c: c["recorded_result"] = m.get("content") or ""; calls.append(c)
    return calls

def canon(cmd):
    """Canonical form for matching: drop a leading `cd .`/`cd /testbed` hop, collapse whitespace."""
    c = re.sub(r"^\s*cd\s+(\.|/testbed/?)\s*(&&|;)\s*", "", (cmd or "").strip())
    return re.sub(r"\s+", " ", c).strip()
def same_cmd(a, b): return a is not None and b is not None and (a.strip() == b.strip() or canon(a) == canon(b))
def is_testish(cmd): return bool(TESTISH.search(cmd)) and not UNSAFE.search(cmd)
def is_readonly(name, args):
    if name == "str_replace_editor": return args.get("command") == "view"
    cmd = (args.get("command") or "").strip()
    return bool(READONLY_BASH.match(cmd)) and not UNSAFE.search(cmd)

def replay_session(sess, inst, image, cname, model_times=(1.5, 6.6, 14.2, 26.2), max_calls=80, per_cmd_timeout=600):
    c = Container(image, cname)
    rec = {"session_id": sess["trajectory_id"], "instance_id": inst["instance_id"], "image": image, "framework": sess.get("framework"), "model": sess.get("model"),
           "calls": [], "spec_events": [], "notes": []}
    try:
        rc, o = c.apply_patch(inst["patch"])
        rec["bug_patch_applied"] = (rc == 0); rec["notes"].append(o[:300])
        calls = parse_calls(sess["trajectory"])[:max_calls]
        fp = c.tree_fingerprint(); status = c.tree_status()
        last_test = None; pending = None   # pending: {cmd, t0}
        for i, call in enumerate(calls):
            name = call["name"]; a = call["args"]; cmd = (a.get("command") or "").strip() if name != "str_replace_editor" else None
            entry = {"i": i, "name": name, "cmd": (cmd or a.get("command")), "readonly": is_readonly(name, a)}
            # --- speculation resolution before executing this call
            if pending and not entry["readonly"]:
                if name != "str_replace_editor" and same_cmd(cmd, pending["cmd"]):
                    # HIT: wait for the speculative run to finish, then measure and validate
                    t_req = time.time()
                    while not c.spec_poll_done():
                        if time.time() - pending["t0"] > per_cmd_timeout: c.spec_kill(); break
                        time.sleep(0.5)
                    t_done = time.time(); D = t_done - pending["t0"]; wait = max(0.0, t_done - t_req)
                    spec_rc, spec_out = c.spec_result()
                    rc, outp, dt = c.exec(cmd, timeout=per_cmd_timeout)   # validation run
                    same = norm(spec_out) == norm(outp)
                    same_sorted = sorted(norm(spec_out).splitlines()) == sorted(norm(outp).splitlines())   # stdout/stderr interleaving differs between capture paths
                    ev = {"i": i, "kind": "hit", "cmd": cmd, "match": "exact" if cmd.strip() == pending["cmd"].strip() else "canonical", "spec_duration_s": D, "real_duration_s": dt, "output_equal_normalized": same,
                          "output_equal_exact": spec_out == outp, "output_equal_lines_sorted": same_sorted, "spec_rc": spec_rc, "real_rc": rc, "edit_kind": pending.get("edit_kind"), "saved_under_model_time": {str(m): min(D, m) for m in model_times}}
                    if not same:
                        import difflib
                        ev["diff_sample"] = "\n".join(list(difflib.unified_diff(norm(spec_out).splitlines(), norm(outp).splitlines(), "spec", "real", lineterm="", n=0))[:40])[:3000]
                    rec["spec_events"].append(ev)
                    entry.update({"rc": rc, "duration_s": dt, "out_chars": len(outp), "spec": "hit"})
                    rec["calls"].append(entry); pending = None
                    fp = c.tree_fingerprint(); last_test = cmd
                    continue
                else:
                    c.spec_kill()
                    rec["spec_events"].append({"i": i, "kind": "miss", "cmd": pending["cmd"], "next": entry["cmd"], "wasted_s": time.time() - pending["t0"], "edit_kind": pending.get("edit_kind")})
                    pending = None
            # --- execute the real call
            if name == "str_replace_editor":
                rc, outp, dt = c.editor(a)
            else:
                rc, outp, dt = c.exec(cmd, timeout=per_cmd_timeout)
            entry.update({"rc": rc, "duration_s": dt, "out_chars": len(outp), "testish": bool(cmd and is_testish(cmd))})
            rec["calls"].append(entry)
            if cmd and is_testish(cmd): last_test = cmd
            # --- did the tree change? (edit detection independent of tool type)
            if not entry["readonly"]:
                nfp = c.tree_fingerprint()
                changed = nfp != fp; fp = nfp
                entry["changed_tree"] = changed
                if changed:
                    nstatus = c.tree_status(); added = nstatus - status; status = nstatus
                    kinds = set(("created" if l.startswith("??") or l.startswith("A ") else "modified") for l in added) or {"modified"}
                    entry["edit_kind"] = "created" if kinds == {"created"} else ("modified" if kinds == {"modified"} else "mixed")
                if changed and last_test and not (cmd and same_cmd(cmd, last_test)) and rc in (0, None) or (changed and last_test and name == "str_replace_editor" and rc == 0):
                    # launch speculation
                    t0 = c.spec_start(last_test); pending = {"cmd": last_test, "t0": t0, "edit_kind": entry.get("edit_kind")}
                    rec["spec_events"].append({"i": i, "kind": "launch", "cmd": last_test, "edit_kind": entry.get("edit_kind")})
        if pending:
            c.spec_kill(); rec["spec_events"].append({"i": len(calls), "kind": "miss", "cmd": pending["cmd"], "next": "<end>", "wasted_s": time.time() - pending["t0"]})
    finally:
        c.close()
    ev = rec["spec_events"]
    rec["summary"] = {"calls": len(rec["calls"]), "launches": sum(e["kind"] == "launch" for e in ev), "hits": sum(e["kind"] == "hit" for e in ev), "misses": sum(e["kind"] == "miss" for e in ev),
                      "hit_durations_s": [e["spec_duration_s"] for e in ev if e["kind"] == "hit"], "hits_output_equal_norm": sum(1 for e in ev if e["kind"] == "hit" and e["output_equal_normalized"]),
                      "total_tool_s": sum(x.get("duration_s", 0) for x in rec["calls"]), "testish_s": sum(x.get("duration_s", 0) for x in rec["calls"] if x.get("testish")),
                      "saved_under_model_time": {str(m): sum(e["saved_under_model_time"][str(m)] for e in ev if e["kind"] == "hit") for m in model_times}}
    return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True); ap.add_argument("--instances", required=True); ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=1000); ap.add_argument("--worker", default="w0"); ap.add_argument("--max-calls", type=int, default=80)
    args = ap.parse_args()
    sessions = json.load(open(args.sessions)); instances = json.load(open(args.instances))
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out): done.add(json.loads(line)["session_id"])
    n = 0
    for s in sessions:
        iid = s.get("instance_id"); inst = instances.get(iid)
        if not inst or inst["image_name"] != args.image or s["trajectory_id"] in done: continue
        t0 = time.time()
        try:
            rec = replay_session(s, inst, args.image, f"spec_{args.worker}", max_calls=args.max_calls)
        except Exception as e:
            rec = {"session_id": s["trajectory_id"], "instance_id": iid, "error": repr(e)[:500]}
        rec["wall_s"] = time.time() - t0
        open(args.out, "a").write(json.dumps(rec) + "\n")
        print(f"[{args.worker}] {iid} calls={rec.get('summary',{}).get('calls')} hits={rec.get('summary',{}).get('hits')} misses={rec.get('summary',{}).get('misses')} wall={rec['wall_s']:.0f}s err={rec.get('error','')[:80]}", flush=True)
        n += 1
        if n >= args.limit: break
