#!/usr/bin/env python3
"""Live coding-agent harness for the phase-1 experiment.

An open model served by vLLM on Modal (OpenAI-compatible, tool calling) drives bash and editor tools inside
Modal Sandboxes built from the official SWE-bench task images. Arms:
  --spec off       vanilla: every tool call runs when the model asks for it
  --spec harness   the two measured rules: after modifying an existing file, pre-execute the last test-like
                   command; after creating a .py file, pre-execute it. A matching next call is served from the
                   speculative run; anything else kills it.
Per task: wall-clock, model time, tool time, sandbox boot, tokens, launches/hits/misses/saved seconds, and the model
patch. Tasks run concurrently (--concurrency) against the one server so tasks per GPU-hour is measured under load.
--mode grade then grades a finished file with the official SWE-bench eval script and log parser (swebench 4.1.0) in
fresh sandboxes, so grading never shares the clock with the timed arms.
"""
import argparse, json, os, re, shlex, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import modal
import urllib.error
import urllib.request
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from replay import is_testish, is_readonly, same_cmd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PREFIX = "swebench/sweb.eval.x86_64."
MAX_TOOL_OUT = 12000

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a bash command in the repository container (cwd /testbed). Commands time out after 300 s.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "str_replace_editor", "description": "View, create, and edit files. command=view (path, optional view_range [start,end]); create (path, file_text); str_replace (path, old_str, new_str; old_str must match exactly once); insert (path, insert_line, new_str).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]}, "path": {"type": "string"}, "file_text": {"type": "string"},
                    "old_str": {"type": "string"}, "new_str": {"type": "string"}, "insert_line": {"type": "integer"}, "view_range": {"type": "array", "items": {"type": "integer"}}}, "required": ["command", "path"]}}},
    {"type": "function", "function": {"name": "submit", "description": "Call when the fix is complete and your reproduction script and the relevant tests pass.", "parameters": {"type": "object", "properties": {}}}},
]
SYSTEM = """You are an expert software engineer fixing a GitHub issue in a Python repository checked out at /testbed inside a container.
Method: find the relevant code, write a small script that reproduces the issue, make the minimal fix, re-run your reproduction script and the relevant existing tests, then call submit.
Use the bash tool for commands and str_replace_editor for viewing and editing files. Be purposeful; do not narrate at length."""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def wait_for_server(base_url, api_key, max_wait=1800):
    """Block until the vLLM endpoint answers /v1/models (covers the cold start of the GPU container)."""
    t0 = time.time(); last = None
    while time.time() - t0 < max_wait:
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:
                    return time.time() - t0
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} (the server is up; a 401 here means a key mismatch, or stale requests queued by an earlier run being flushed)"
        except Exception as e:
            last = repr(e)[:200]
        time.sleep(10)
    raise RuntimeError(f"server not ready after {max_wait}s: {last}")


class SandboxEnv:
    """Same interface as replay.Container, backed by a Modal Sandbox from the task's official image."""

    def __init__(self, image_ref, app, workdir="/testbed", timeout_s=3 * 3600):
        img = modal.Image.from_registry(image_ref)
        self.sb = modal.Sandbox.create(image=img, app=app, timeout=timeout_s, workdir=workdir, cpu=2, memory=8192)
        self.wd = workdir
        with open(os.path.join(HERE, "..", "editor.py")) as f:
            editor_src = f.read()
        self.write("/tmp/editor.py", editor_src)

    def write(self, path, content):
        """Sandbox.open() was removed from Modal in 2026; use the filesystem API, falling back to exec + stdin."""
        try:
            self.sb.filesystem.write_text(path, content)
        except Exception:
            rc, out, _ = self.exec(f"cat > {shlex.quote(path)}", timeout=60, stdin=content)
            if rc != 0:
                raise RuntimeError(f"could not write {path}: {out[-300:]}")

    def exec(self, cmd, timeout=300, stdin=None):
        t0 = time.time()
        wrapped = f"timeout -k 5 {int(timeout)} bash -lc {shlex.quote(cmd)}"
        p = self.sb.exec("bash", "-lc", wrapped, workdir=self.wd, timeout=timeout + 30)
        if stdin is not None:
            p.stdin.write(stdin); p.stdin.write_eof(); p.stdin.drain()
        out = p.stdout.read(); err = p.stderr.read(); rc = p.wait()
        return rc, (out + err), time.time() - t0

    def tree_fingerprint(self):
        _, o, _ = self.exec("git status --porcelain=v1 2>/dev/null | md5sum; git diff 2>/dev/null | md5sum", timeout=60)
        return o.strip()

    def tree_status(self):
        _, o, _ = self.exec("git status --porcelain=v1 2>/dev/null | head -200", timeout=60)
        return set(l for l in o.splitlines() if l.strip())

    def spec_start(self, cmd):
        inner = "timeout -k 5 300 bash -c " + shlex.quote(cmd) + "; echo $? > /tmp/spec.rc"
        wrapped = f"rm -f /tmp/spec.rc /tmp/spec.out; setsid bash -c {shlex.quote(inner)} > /tmp/spec.out 2>&1 < /dev/null & echo $! > /tmp/spec.pid"
        self.exec(wrapped, timeout=30)
        return time.time()

    def spec_poll_done(self):
        _, o, _ = self.exec("test -f /tmp/spec.rc && echo done || echo running", timeout=30)
        return "done" in o

    def spec_kill(self):
        self.exec("if [ -f /tmp/spec.pid ]; then kill -TERM -- -$(cat /tmp/spec.pid) 2>/dev/null; sleep 0.2; kill -KILL -- -$(cat /tmp/spec.pid) 2>/dev/null; fi; rm -f /tmp/spec.pid", timeout=30)

    def spec_result(self):
        _, o, _ = self.exec("cat /tmp/spec.rc; echo ---SPECOUT---; cat /tmp/spec.out", timeout=60)
        head, _, body = o.partition("---SPECOUT---\n")
        return head.strip(), body

    def editor(self, args):
        self.write("/tmp/editor_in.json", json.dumps(args))
        return self.exec("python3 /tmp/editor.py < /tmp/editor_in.json", timeout=120)

    def close(self):
        try:
            self.sb.terminate()
        except Exception:
            pass


def predicted_after_create(path):
    """Rule B target for a newly created Python file. Predicted with a path relative to /testbed, which is how the
    model invokes its own scripts; the matcher below also treats /testbed/x and x as the same file."""
    if path and path.endswith(".py"):
        rel = path[len("/testbed/"):] if path.startswith("/testbed/") else path
        return f"cd /testbed && python {rel}"
    return None


_TESTBED_PREFIX = re.compile(r"(?<![\w/])/testbed/")


def same_cmd_here(a, b):
    """same_cmd from the replay engine, plus: paths under /testbed compare equal to their relative form."""
    if a is None or b is None:
        return False
    return same_cmd(a, b) or same_cmd(_TESTBED_PREFIX.sub("", a), _TESTBED_PREFIX.sub("", b))


def clip(s, n=MAX_TOOL_OUT):
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


def shrink_history(messages, keep_last=6):
    """Elide old tool outputs when the context overflows; returns True if anything changed."""
    idx = [i for i, m in enumerate(messages) if m["role"] == "tool" and len(m["content"]) > 200]
    changed = False
    for i in idx[:-keep_last] if len(idx) > keep_last else []:
        messages[i]["content"] = "[earlier output elided to fit the context window]"; changed = True
    return changed


def chat(client, model, messages):
    last = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.2, max_tokens=4096)
            if not hasattr(resp, "choices") or not resp.choices:
                # the OpenAI client hands back the raw body when the gateway answers with something that is not JSON
                last = RuntimeError(f"non-completion response: {str(resp)[:300]!r}")
                print(f"model call returned a non-completion body (attempt {attempt + 1}): {str(resp)[:200]!r}", flush=True)
                time.sleep(3 * (attempt + 1)); continue
            return resp
        except Exception as e:
            last = e; msg = str(e)
            if ("maximum context length" in msg or "context_length" in msg or "too long" in msg) and shrink_history(messages):
                continue
            time.sleep(3 * (attempt + 1))
    raise last


def run_task(inst, client, model, spec, app, max_steps=40, per_cmd_timeout=300):
    iid = inst["instance_id"]
    image = f"{IMAGE_PREFIX}{iid.replace('__', '_1776_')}:latest"
    rec = {"instance_id": iid, "repo": inst["repo"], "model": model, "arm": "harness" if spec else "off", "image": image,
           "started_at": now_iso(), "steps": [], "spec_events": [], "tokens": {"prompt": 0, "completion": 0}}
    t_start = time.time(); model_s = 0.0; tool_s = 0.0; env = None; submitted = False
    try:
        env = SandboxEnv(image, app)
        rec["sandbox_boot_s"] = time.time() - t_start
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"<issue>\n{inst['problem_statement']}\n</issue>\nFix this issue in the repository at /testbed."}]
        fp = env.tree_fingerprint(); status = env.tree_status(); last_test = None; pending = None
        for step in range(max_steps):
            t0 = time.time()
            resp = chat(client, model, messages)
            dt_model = time.time() - t0; model_s += dt_model
            u = resp.usage
            rec["tokens"]["prompt"] += u.prompt_tokens; rec["tokens"]["completion"] += u.completion_tokens
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            else:
                messages.append({"role": "assistant", "content": msg.content or ""})
                rec["steps"].append({"step": step, "model_s": dt_model, "end": resp.choices[0].finish_reason})
                break
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except Exception:
                    a = {"_raw": tc.function.arguments}
                cmd = (a.get("command") or "").strip() if name == "bash" else None
                entry = {"step": step, "model_s": dt_model, "tool": name, "cmd": cmd or a.get("command"),
                         "path": a.get("path") if name == "str_replace_editor" else None,
                         "readonly": is_readonly("str_replace_editor" if name == "str_replace_editor" else "bash", a)}
                if name == "submit":
                    submitted = True
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "submitted"}); rec["steps"].append(entry)
                    break
                served = None
                if pending and not entry["readonly"]:
                    if name == "bash" and same_cmd_here(cmd, pending["cmd"]):
                        t_req = time.time()
                        while not env.spec_poll_done():
                            if time.time() - pending["t0"] > per_cmd_timeout:
                                env.spec_kill(); break
                            time.sleep(0.3)
                        t_done = time.time(); D = t_done - pending["t0"]; wait = max(0.0, t_done - t_req)
                        spec_rc, spec_out = env.spec_result()
                        served = (int(spec_rc) if spec_rc.strip().isdigit() else 1, spec_out, wait)
                        rec["spec_events"].append({"step": step, "kind": "hit", "rule": pending["rule"], "cmd": cmd, "spec_duration_s": D, "waited_s": wait, "saved_s": max(0.0, D - wait)})
                        pending = None
                    else:
                        env.spec_kill()
                        rec["spec_events"].append({"step": step, "kind": "miss", "rule": pending["rule"], "cmd": pending["cmd"], "next": entry["cmd"], "wasted_s": time.time() - pending["t0"]})
                        pending = None
                if served:
                    rc, outp, dt = served
                elif name == "bash":
                    rc, outp, dt = env.exec(cmd, timeout=per_cmd_timeout)
                else:
                    rc, outp, dt = env.editor(a)
                tool_s += dt
                outp = clip(outp)
                entry.update({"rc": rc, "tool_s": dt, "out_chars": len(outp), "served_from_spec": bool(served)})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": outp or "(no output)"})
                if cmd and is_testish(cmd):
                    last_test = cmd
                if not entry["readonly"]:
                    nfp = env.tree_fingerprint(); changed = nfp != fp; fp = nfp; entry["changed_tree"] = changed
                    if changed:
                        nstatus = env.tree_status(); added = nstatus - status; status = nstatus
                        def kind_of(l):
                            if l.startswith("??") or l.startswith("A "): return "created"
                            if l.startswith(" D") or l.startswith("D "): return "deleted"
                            return "modified"
                        kinds = set(kind_of(l) for l in added) or {"modified"}
                        entry["edit_kind"] = next(iter(kinds)) if len(kinds) == 1 else "mixed"
                        if spec and rc in (0, None):
                            target = None; rule = None
                            if entry["edit_kind"] == "modified" and last_test and not (cmd and same_cmd_here(cmd, last_test)):
                                target, rule = last_test, "A"
                            elif entry["edit_kind"] == "created":
                                m = re.search(r">\s*(\S+\.py)\b", cmd or "")
                                created = entry["path"] or (m.group(1) if m else None)
                                target = predicted_after_create(created); rule = "B"
                            if target:
                                pending = {"cmd": target, "t0": env.spec_start(target), "rule": rule}
                                rec["spec_events"].append({"step": step, "kind": "launch", "rule": rule, "cmd": target})
                rec["steps"].append(entry)
            if submitted:
                break
        if pending:
            env.spec_kill()
        rec["agent_s"] = time.time() - t_start
        _, diff, _ = env.exec("git add -A -N . && git diff -- . && git reset -q", timeout=60)
        rec["model_patch"] = diff[:200000]
    except Exception as e:
        rec["error"] = repr(e)[:800]
    finally:
        if env:
            env.close()
    rec.update({"ended_at": now_iso(), "wall_s": time.time() - t_start, "model_s": model_s, "tool_s": tool_s, "submitted": submitted,
                "launches": sum(e["kind"] == "launch" for e in rec["spec_events"]),
                "hits": sum(e["kind"] == "hit" for e in rec["spec_events"]),
                "misses": sum(e["kind"] == "miss" for e in rec["spec_events"]),
                "saved_s": sum(e.get("saved_s", 0) for e in rec["spec_events"]),
                "wasted_s": sum(e.get("wasted_s", 0) for e in rec["spec_events"])})
    return rec


def grade(inst, model_patch, model, app):
    """Official SWE-bench grading in a fresh sandbox: apply the model patch, run the harness eval script, parse
    with the repo's log parser. Runs after the timed arms so grading never shares the clock with the agent loop."""
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.grading import get_eval_report
    iid = inst["instance_id"]
    if not model_patch.strip():
        return {"resolved": False, "patch_applied": False, "note": "empty patch"}
    spec = make_test_spec(inst)
    image = f"{IMAGE_PREFIX}{iid.replace('__', '_1776_')}:latest"
    env = None; t0 = time.time()
    try:
        env = SandboxEnv(image, app, timeout_s=3600)
        env.write("/tmp/model.patch", model_patch)
        rc_apply, out_apply, _ = env.exec("git apply -v /tmp/model.patch || git apply -v --reject /tmp/model.patch || patch --batch --fuzz=5 -p1 -i /tmp/model.patch", timeout=120)
        if rc_apply != 0:
            return {"resolved": False, "patch_applied": False, "duration_s": time.time() - t0, "tail": out_apply[-1500:]}
        env.write("/tmp/eval.sh", spec.eval_script)
        rc, out, _ = env.exec("bash /tmp/eval.sh", timeout=1800)
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(out); log_path = f.name
        try:
            report = get_eval_report(spec, {"instance_id": iid, "model_name_or_path": model, "model_patch": model_patch}, log_path, True)
        finally:
            os.unlink(log_path)
        r = report[iid]
        ts = r.get("tests_status", {}) or {}
        counts = {k: {"success": len(v.get("success", [])), "failure": len(v.get("failure", []))} for k, v in ts.items()}
        return {"resolved": bool(r.get("resolved")), "patch_applied": True, "rc": rc, "duration_s": time.time() - t0, "tests": counts, "tail": out[-1500:]}
    except Exception as e:
        return {"resolved": False, "error": repr(e)[:400], "duration_s": time.time() - t0}
    finally:
        if env:
            env.close()


def grade_file(path, tasks_by_id, model, app, concurrency):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    todo = [r for r in recs if "grade" not in r and not r.get("error")]
    print(f"grading {len(todo)} of {len(recs)} records in {path} (concurrency {concurrency})", flush=True)
    lock = threading.Lock()

    def one(r):
        g = grade(tasks_by_id[r["instance_id"]], r.get("model_patch", ""), model, app)
        with lock:
            r["grade"] = g
        print(f"{r['instance_id']} arm={r.get('arm')} resolved={g.get('resolved')} applied={g.get('patch_applied')} tests={g.get('tests')} err={g.get('error', '')}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, todo))
    for r in recs:
        if r.get("error") and "grade" not in r:
            r["grade"] = {"resolved": False, "note": "task errored"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks_verified_50.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--model", default=os.environ.get("SPEC_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"))
    ap.add_argument("--mode", choices=["run", "grade"], default="run")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=os.environ.get("SPEC_VLLM_API_KEY", "spec-exec-local"))
    ap.add_argument("--spec", choices=["off", "harness", "both"], default="harness",
                    help="both: run every task in both arms in the same worker, alternating which arm goes first, and write <out>.off.jsonl / <out>.harness.jsonl")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    all_tasks = json.load(open(args.tasks))
    app = modal.App.lookup("spec-exec-sandboxes", create_if_missing=True)
    if args.mode == "grade":
        grade_file(args.out, {t["instance_id"]: t for t in all_tasks}, args.model, app, args.concurrency)
        return
    if not args.base_url:
        ap.error("--base-url is required in run mode")
    tasks = all_tasks[args.start:args.start + args.count]
    interleave = args.spec == "both"
    outs = {"off": args.out + ".off.jsonl", "harness": args.out + ".harness.jsonl"} if interleave else {args.spec: args.out}
    done = set()
    for path in outs.values():
        if os.path.exists(path):
            for line in open(path):
                try:
                    done.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
    tasks = [t for t in tasks if t["instance_id"] not in done]
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key=args.api_key, timeout=900, max_retries=2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ready_s = wait_for_server(args.base_url, args.api_key)
    print(f"server ready after {ready_s:.0f}s; arm={args.spec} tasks={len(tasks)} concurrency={args.concurrency}", flush=True)
    lock = threading.Lock(); run_t0 = time.time()

    def emit(rec, arm):
        rec["concurrency"] = args.concurrency; rec["run_elapsed_s"] = time.time() - run_t0
        with lock:
            with open(outs[arm], "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"{rec['instance_id']} arm={arm} wall={rec['wall_s']:.0f}s model={rec['model_s']:.0f}s tool={rec['tool_s']:.0f}s "
              f"steps={len(rec['steps'])} hits={rec['hits']} misses={rec['misses']} saved={rec['saved_s']:.0f}s tokens={rec['tokens']} "
              f"patch_chars={len(rec.get('model_patch', ''))} err={rec.get('error', '')[:120]}", flush=True)

    def job(idx, inst):
        arms = ["off", "harness"] if not interleave else (["off", "harness"] if idx % 2 == 0 else ["harness", "off"])
        if not interleave:
            arms = [args.spec]
        for arm in arms:
            try:
                rec = run_task(inst, client, args.model, arm == "harness", app, args.max_steps)
            except Exception as e:
                rec = {"instance_id": inst["instance_id"], "arm": arm, "error": repr(e)[:800], "wall_s": 0, "model_s": 0, "tool_s": 0, "steps": [], "spec_events": [], "tokens": {}, "hits": 0, "misses": 0, "saved_s": 0}
            rec["arm_order"] = arms.index(arm)
            emit(rec, arm)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(lambda p: job(*p), list(enumerate(tasks))))
    print(f"spec={args.spec} done: {len(tasks)} tasks in {time.time() - run_t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
