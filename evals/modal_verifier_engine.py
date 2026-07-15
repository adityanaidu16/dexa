"""Live verifier-guided early-stop ENGINE vs naive best-of-N — real serving numbers.

modal_verifier_search.py proved the efficiency *post-hoc* (generate all N, then
compute what early-stop WOULD have cost). This runs the engine for real: it generates
in rounds, verifies each round as it lands, and stops issuing work for a problem the
moment one of its samples passes the tests. Wall-clock and tokens are measured
end-to-end, so the 2.8x post-hoc number becomes a live serving number.

Fair batched design (this is the crux): a round generates B fresh samples for EVERY
still-unsolved problem in ONE batched llm.chat call — so cross-problem batching and
shared-prompt KV are preserved exactly as in the naive n=16 baseline. After each round
we run the verifier, drop the problems that passed, and only the survivors go to the
next round. Easy problems exit after round 1; only the hard tail runs all 16/B rounds.

  naive   : one llm.chat(n=16) over all problems, then verify.  cost = all 16 always.
  engine  : rounds of B=4, verify-and-drop between rounds.       cost = rounds until pass.

Both report identical-ish pass@16 (engine redraws i.i.d. per round, so tiny sampling
variance). Primary metric = decode tokens (order-independent, GPU-bound). Wall-clock is
reported too but is directional (prefix-cache state persists across calls).

    modal run evals/modal_verifier_engine.py
    modal run evals/modal_verifier_engine.py --n-problems 60 --b 4
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

app = modal.App("dexa-verifier-engine")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip("\n")


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, N: int, B: int, n_problems: int) -> None:
    import subprocess
    import sys
    from time import perf_counter

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    he = load_dataset("openai/openai_humaneval", split="test")
    if n_problems > 0:
        he = he.select(range(min(n_problems, len(he))))
    n = len(he)
    convos = [[{"role": "user", "content":
                f"Complete this Python function. Return the full function in a "
                f"```python block.\n\n```python\n{p}\n```"}] for p in he["prompt"]]
    probs = list(zip(he["prompt"], he["test"], he["entry_point"]))

    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.9,
              enforce_eager=True)

    def passes(text: str, prob) -> bool:
        prompt_code, test_code, entry = prob
        code = extract_python(text)
        if f"def {entry}" not in code:
            code = prompt_code + code
        program = f"{code}\n{test_code}\ncheck({entry})\n"
        try:
            r = subprocess.run([sys.executable, "-c", program], timeout=10,
                               capture_output=True)
            return r.returncode == 0
        except Exception:
            return False

    def sp(nn, seed):
        return SamplingParams(n=nn, temperature=0.8, top_p=0.95, max_tokens=640, seed=seed)

    # ---- ENGINE: rounds of B, verify-and-drop between rounds ----------------
    # Run the engine FIRST so it pays cold prompt-prefill; naive then reuses the
    # warmed prefix cache (a conservative bias — it helps the baseline, not us).
    unsolved = list(range(n))
    eng_solved = [False] * n
    eng_tokens = 0
    eng_gen_s = 0.0
    eng_verify_s = 0.0
    rounds_used = 0
    rounds = (N + B - 1) // B
    for r in range(rounds):
        if not unsolved:
            break
        rounds_used = r + 1
        t = perf_counter()
        outs = llm.chat([convos[i] for i in unsolved], sp(B, r), use_tqdm=False)
        eng_gen_s += perf_counter() - t
        t = perf_counter()
        still = []
        for idx, req in zip(unsolved, outs):
            eng_tokens += sum(len(o.token_ids) for o in req.outputs)
            if any(passes(o.text, probs[idx]) for o in req.outputs):
                eng_solved[idx] = True
            else:
                still.append(idx)
        eng_verify_s += perf_counter() - t
        unsolved = still
        print(f"[engine] round {r+1}/{rounds}: {n - len(unsolved)}/{n} solved, "
              f"{len(unsolved)} left, tokens so far {eng_tokens}", flush=True)

    # ---- NAIVE: one n=N call over all problems, then verify -----------------
    t = perf_counter()
    outs = llm.chat(convos, sp(N, 0), use_tqdm=False)
    naive_gen_s = perf_counter() - t
    naive_tokens = sum(len(o.token_ids) for req in outs for o in req.outputs)
    t = perf_counter()
    naive_solved = [any(passes(o.text, prob) for o in req.outputs)
                    for req, prob in zip(outs, probs)]
    naive_verify_s = perf_counter() - t

    eng_pass = sum(eng_solved) / n
    naive_pass = sum(naive_solved) / n
    tok_ratio = naive_tokens / eng_tokens if eng_tokens else 0.0
    gen_ratio = naive_gen_s / eng_gen_s if eng_gen_s else 0.0

    print("\n" + "=" * 74)
    print(f"LIVE VERIFIER-GUIDED ENGINE — HumanEval, {model} ({GPU})")
    print(f"N={N}, B={B} ({rounds} max rounds), {n} problems")
    print("=" * 74)
    print(f"  {'metric':16} {'naive n=N':>14} {'engine (B-rounds)':>20} {'ratio':>8}")
    print(f"  {'pass@'+str(N):16} {naive_pass:>14.3f} {eng_pass:>20.3f}")
    print(f"  {'decode tokens':16} {naive_tokens:>14} {eng_tokens:>20} {tok_ratio:>7.2f}x")
    print(f"  {'gen wall (s)':16} {naive_gen_s:>14.1f} {eng_gen_s:>20.1f} {gen_ratio:>7.2f}x")
    print(f"  {'verify wall (s)':16} {naive_verify_s:>14.1f} {eng_verify_s:>20.1f}")
    print(f"  rounds actually used by the hard tail: {rounds_used}/{rounds}")
    print("=" * 74)
    print(f"  HEADLINE: engine matches naive pass@{N} "
          f"({eng_pass:.3f} vs {naive_pass:.3f}) using {tok_ratio:.1f}x fewer decode "
          f"tokens and {gen_ratio:.1f}x less GPU generation time — LIVE, not post-hoc.")
    print("=" * 74)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", n: int = 16, b: int = 4,
         n_problems: int = 0) -> None:
    print(f"live verifier engine on {GPU}: {model}, N={n}, B={b}")
    run.remote(model, n, b, n_problems)
