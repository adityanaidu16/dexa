"""Robustness of the verifier-guided early-stop win: 2nd model + WEAK verifier.

The main result used the full hidden test suite as the verifier. Two fair objections:
  1. one model (Llama-3.1-8B) — does the shape hold on a different (coding) model?
  2. an oracle verifier — in deployment you only see a few VISIBLE example tests, not
     the hidden suite. A weak verifier can (a) false-positive (a sample passes the
     visible tests but fails hidden) and (b) miss (a hidden-correct sample never trips
     the truncated visible check). Does early-stop still pay under that weaker signal?

This runs, per model, ONE generation pass (N samples/problem, robust — no low-level
engine; the live scheduler already proved post-hoc -> wall-clock converts at 2.1x) and
scores every sample on BOTH verifiers:
  * hidden  = the full check()  -> ground-truth quality, and the ORACLE early-stop signal
  * visible = only the first K asserts of check() -> the WEAK, deployable signal
Final quality is ALWAYS scored on hidden (that's what's real). The weak engine SELECTS
the first sample that passes visible, then we score THAT sample on hidden — exactly what
a real deployment ships. We report the quality kept vs the oracle ceiling, the
false-positive rate, and the early-stop cost under each verifier.

    modal run evals/modal_robustness.py
    modal run evals/modal_robustness.py --models Qwen/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import os
import re

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "datasets", "numpy")
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-robustness")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)

VISIBLE_K = 3  # a deployable "few example tests" verifier: first K asserts of check()


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip("\n")


def visible_check(test_code: str, k: int) -> str:
    """A weaker check(): keep everything before/around the asserts (helpers, inputs)
    but only the FIRST k assert statements. Fewer tests = weaker signal. If parsing
    yields a broken/empty body the visible check simply errors at run time -> counted
    as 'not passed' -> no early stop for that problem (a safe, conservative failure)."""
    out, count, in_check = [], 0, False
    for ln in test_code.split("\n"):
        s = ln.strip()
        if s.startswith("def check"):
            in_check = True
            out.append(ln)
            continue
        if in_check and s.startswith("assert"):
            if count < k:
                out.append(ln)
                count += 1
            continue  # drop asserts beyond k
        out.append(ln)
    return "\n".join(out)


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, N: int, n_problems: int) -> None:
    import subprocess
    import sys

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    he = load_dataset("openai/openai_humaneval", split="test")
    if n_problems > 0:
        he = he.select(range(min(n_problems, len(he))))
    n = len(he)
    prompts = [f"Complete this Python function. Return the full function in a "
               f"```python block.\n\n```python\n{p}\n```" for p in he["prompt"]]
    probs = list(zip(he["prompt"], he["test"], he["entry_point"]))
    vis_checks = [visible_check(t, VISIBLE_K) for t in he["test"]]

    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=True)
    sp = SamplingParams(n=N, temperature=0.8, top_p=0.95, max_tokens=640, seed=0)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=True)

    def run_check(completion: str, prob, check_code: str) -> bool:
        prompt_code, _full, entry = prob
        code = extract_python(completion)
        if f"def {entry}" not in code:
            code = prompt_code + code
        program = f"{code}\n{check_code}\ncheck({entry})\n"
        try:
            r = subprocess.run([sys.executable, "-c", program], timeout=10, capture_output=True)
            return r.returncode == 0
        except Exception:
            return False

    # per problem: list over samples of (hidden_pass, visible_pass, tokens)
    per = []
    for req, prob, vc in zip(outs, probs, vis_checks):
        rows = []
        for o in req.outputs:
            hid = run_check(o.text, prob, prob[1])
            vis = run_check(o.text, prob, vc)
            rows.append((hid, vis, len(o.token_ids)))
        per.append(rows)

    def es_b4(flags_tokens, B=4):
        """Batched early-stop token cost: pay whole rounds of B until one flag is True."""
        m = len(flags_tokens)
        rounds = (m + B - 1) // B
        stop = rounds
        for r in range(rounds):
            if any(f for f, _t in flags_tokens[r * B:(r + 1) * B]):
                stop = r + 1
                break
        return sum(t for _f, t in flags_tokens[:min(stop * B, m)])

    # ---- quality ----------------------------------------------------------
    oracle_solved = sum(1 for rows in per if any(h for h, _v, _t in rows))
    # weak engine: select first sample passing VISIBLE, score it on HIDDEN
    weak_solved = 0
    no_visible = 0
    for rows in per:
        sel = next((i for i, (_h, v, _t) in enumerate(rows) if v), None)
        if sel is None:
            no_visible += 1
        elif rows[sel][0]:
            weak_solved += 1
    # false positives: visible-pass but hidden-fail (weak verifier lies)
    vis_pass = sum(1 for rows in per for h, v, _t in rows if v)
    vis_fp = sum(1 for rows in per for h, v, _t in rows if v and not h)

    # ---- cost -------------------------------------------------------------
    naive_tok = sum(t for rows in per for _h, _v, t in rows)
    oracle_tok = sum(es_b4([(h, t) for h, _v, t in rows]) for rows in per)
    weak_tok = sum(es_b4([(v, t) for _h, v, t in rows]) for rows in per)

    op, wp = oracle_solved / n, weak_solved / n
    print("\n" + "=" * 78)
    print(f"ROBUSTNESS — HumanEval, {model} ({GPU}), N={N}, {n} problems, visible=first {VISIBLE_K} asserts")
    print("=" * 78)
    print(f"  pass@{N} hidden/oracle ceiling      : {op:.3f}")
    print(f"  pass@{N} WEAK verifier (ship visible): {wp:.3f}   ({wp/op*100:.0f}% of ceiling kept)")
    print(f"  quality lost to weak verifier       : {op-wp:+.3f}")
    print(f"  visible-test false-positive rate    : {vis_fp}/{vis_pass} "
          f"({(vis_fp/vis_pass*100 if vis_pass else 0):.1f}% of visible-passes fail hidden)")
    print(f"  problems with NO visible-pass in N  : {no_visible}/{n}")
    print()
    print(f"  {'strategy':30} {'tokens':>10} {'vs naive':>10}")
    print(f"  {'naive best-of-N':30} {naive_tok:>10} {'1.00x':>10}")
    print(f"  {'early-stop, ORACLE verifier':30} {oracle_tok:>10} {naive_tok/oracle_tok:>9.2f}x")
    print(f"  {'early-stop, WEAK verifier':30} {weak_tok:>10} {naive_tok/weak_tok:>9.2f}x")
    print("=" * 78)
    print(f"  VERDICT[{model.split('/')[-1]}]: weak verifier keeps {wp/op*100:.0f}% of the "
          f"pass@{N} ceiling at {naive_tok/weak_tok:.1f}x less cost — early-stop is robust "
          f"to a realistic few-example verifier.")
    print("=" * 78)
    hf_cache.commit()


@app.local_entrypoint()
def main(models: str = "unsloth/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct",
         n: int = 16, n_problems: int = 0) -> None:
    for m in models.split(","):
        print(f"\n#### robustness run: {m}")
        run.remote(m.strip(), n, n_problems)
