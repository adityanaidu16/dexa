#!/usr/bin/env python3
"""Minimal str_replace_editor emulation run inside the task container. Reads one JSON request on stdin."""
import json, sys, os
req = json.load(sys.stdin)
cmd = req.get("command"); path = req.get("path") or ""
def out(s): sys.stdout.write(s); sys.exit(0)
def err(s): sys.stdout.write(s); sys.exit(1)
if cmd == "view":
    if os.path.isdir(path):
        items = []
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth > 1: continue
            for d in dirs: items.append(os.path.join(root, d) + "/")
            for f in files: items.append(os.path.join(root, f))
        out("\n".join(sorted(items)[:400]) + "\n")
    if not os.path.isfile(path): err(f"The path {path} does not exist.")
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    rng = req.get("view_range")
    lo, hi = (1, len(lines)) if not rng else (max(1, int(rng[0])), (len(lines) if int(rng[1]) == -1 else min(len(lines), int(rng[1]))))
    body = "\n".join(f"{i:6}\t{lines[i-1]}" for i in range(lo, hi + 1))
    out(f"Here's the result of running `cat -n` on {path}:\n{body}\n")
elif cmd == "create":
    if os.path.exists(path): err(f"File already exists at: {path}. Cannot overwrite files using command `create`.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w", encoding="utf-8").write(req.get("file_text") or "")
    out(f"File created successfully at: {path}")
elif cmd == "str_replace":
    if not os.path.isfile(path): err(f"The path {path} does not exist.")
    text = open(path, encoding="utf-8", errors="replace").read()
    old = req.get("old_str") or ""; new = req.get("new_str") or ""
    n = text.count(old)
    if n == 0: err(f"No replacement was performed, old_str `{old[:80]}` did not appear verbatim in {path}.")
    if n > 1: err(f"No replacement was performed. Multiple occurrences of old_str `{old[:80]}` in lines. Please ensure it is unique.")
    text = text.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(text)
    out(f"The file {path} has been edited.")
elif cmd == "insert":
    if not os.path.isfile(path): err(f"The path {path} does not exist.")
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    at = int(req.get("insert_line") or 0)
    ins = (req.get("new_str") or "").split("\n")
    lines[at:at] = ins
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    out(f"The file {path} has been edited.")
elif cmd == "undo_edit":
    out("undo_edit is not supported in replay")
else:
    err(f"Unrecognized command {cmd}")
