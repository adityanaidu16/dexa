"""Verifier-guided search efficiency: pass@N quality at a FRACTION of naive cost.

The eval (evals/RESULTS.md) showed best-of-N lifts HumanEval pass 0.69->0.90 (+21 pts)
but at ~16x tokens because every problem ran all N samples to completion. This proves
the product claim: a verifier-guided engine reaches the SAME pass@N quality far cheaper
by EARLY-STOPPING — stop generating as soon as a sample passes the tests, and spend the
saved budget nowhere. Easy problems finish in 1 sample; only hard ones use the full N.

Method (exact, one generation pass): generate N samples per problem, record each
sample's pass/fail and token count, then compute costs post-hoc (samples are i.i.d.,
so any draw order is valid):
  * naive best-of-N   : cost = all N samples' tokens for every problem.
  * early-stop (B=1)   : cost = tokens up to the first passing sample (theoretical min).
  * early-stop (B=4)   : realistic batched rounds of 4 — stop after the first round that
                         contains a pass (batches within a round share the prompt KV).
All three yield the IDENTICAL pass@N (a problem is solved iff any of its N pass); only
the cost differs. Headline = naive_tokens / early_stop_tokens at equal quality.

Note on the verifier: for code the unit tests ARE the verifier and are ~free and
reliable, so early-stop is realistic. A weaker verifier (visible tests only) recovers
slightly less. Pruning mid-generation isn't natural for code (can't test half a
function); early-stop across complete samples is the valid lever here.

    modal run evals/modal_verifier_search.py
    modal run evals/modal_verifier_search.py --n 32
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

app = modal.App("dexa-verifier-search")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip("\n")


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, N: int, n_problems: int) -> None:
    import subprocess
    import sys

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    he = load_dataset("openai/openai_humaneval", split="test")
    if n_problems > 0:
        he = he.select(range(min(n_problems, len(he))))
    prompts = [f"Complete this Python function. Return the full function in a "
               f"```python block.\n\n```python\n{p}\n```" for p in he["prompt"]]

    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=True)
    sp = SamplingParams(n=N, temperature=0.8, top_p=0.95, max_tokens=640, seed=0)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=True)

    def passes(completion: str, prob) -> bool:
        prompt_code, test_code, entry = prob
        code = extract_python(completion)
        if f"def {entry}" not in code:
            code = prompt_code + code
        program = f"{code}\n{test_code}\ncheck({entry})\n"
        try:
            r = subprocess.run([sys.executable, "-c", program], timeout=10, capture_output=True)
            return r.returncode == 0
        except Exception:
            return False

    # per problem: list of (passed, tokens) in sample order
    per_problem = []
    probs = list(zip(he["prompt"], he["test"], he["entry_point"]))
    for req, prob in zip(outs, probs):
        rows = [(passes(o.text, prob), len(o.token_ids)) for o in req.outputs]
        per_problem.append(rows)

    def first_pass_idx(rows):
        for i, (p, _t) in enumerate(rows):
            if p:
                return i
        return None

    n = len(per_problem)
    solved = sum(1 for rows in per_problem if first_pass_idx(rows) is not None)
    naive_tokens = sum(t for rows in per_problem for _p, t in rows)

    # early-stop B=1: tokens up to & incl first pass, else all N
    es1 = 0
    samples_used = []
    for rows in per_problem:
        fp = first_pass_idx(rows)
        k = (fp + 1) if fp is not None else len(rows)
        samples_used.append(k)
        es1 += sum(t for _p, t in rows[:k])

    # early-stop B=4: stop after the first round-of-4 containing a pass
    B = 4
    es4 = 0
    for rows in per_problem:
        rounds = (len(rows) + B - 1) // B
        stop = rounds
        for r in range(rounds):
            if any(p for p, _t in rows[r * B:(r + 1) * B]):
                stop = r + 1
                break
        es4 += sum(t for _p, t in rows[:stop * B])

    passk = solved / n
    print("\n" + "=" * 72)
    print(f"VERIFIER-GUIDED SEARCH — HumanEval, {model} ({GPU}), N={N}, {n} problems")
    print("=" * 72)
    print(f"  pass@{N} (identical for all strategies): {passk:.3f}")
    print(f"  avg samples used (early-stop B=1): {sum(samples_used)/n:.2f} / {N}")
    print()
    print(f"  {'strategy':22} {'tokens':>10} {'vs naive':>10}  (same {passk:.3f} pass)")
    print(f"  {'naive best-of-N':22} {naive_tokens:>10} {'1.00x':>10}")
    print(f"  {'early-stop B=1 (min)':22} {es1:>10} {naive_tokens/es1:>9.2f}x")
    print(f"  {'early-stop B=4 (batched)':22} {es4:>10} {naive_tokens/es4:>9.2f}x")
    print("=" * 72)
    print(f"  HEADLINE: pass@{N}={passk:.3f} quality at {naive_tokens/es4:.1f}x less cost "
          f"(batched early-stop) — same quality, fewer tokens.")
    print("=" * 72)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", n: int = 16, n_problems: int = 0) -> None:
    print(f"verifier-guided search efficiency on {GPU}: {model}, N={n}")
    run.remote(model, n, n_problems)
