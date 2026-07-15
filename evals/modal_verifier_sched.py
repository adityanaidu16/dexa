"""Continuous-batching verifier-guided scheduler — close the wall-clock gap.

modal_verifier_engine.py proved the LIVE token saving (2.6x) but only got 1.5x
wall-clock, because it ran DISCRETE GLOBAL ROUNDS: round r+1 can't start until every
problem finished round r, so the batch drains to a handful of hard problems and the
A100 idles. This removes the global barrier.

It drives vLLM's low-level LLMEngine step loop directly (add_request / step / abort),
so every problem's samples live in ONE continuously-batched flight with no rounds:
  * seed the engine with B samples for all N problems at once (saturate immediately),
  * on each step, verify whichever samples just finished,
  * the moment one sample for a problem passes, ABORT its still-decoding siblings and
    stop launching more for it (free the KV/compute instantly),
  * a problem whose whole in-flight wave failed launches its next B (up to N), and that
    launch drops straight into the same live batch — no waiting on other problems.
Hard problems (still on their 3rd wave) and easy ones (just launched) decode together,
so the GPU stays full end to end. Abort converts the token saving into wall-clock.

Baseline: the SAME engine run naively (one n=N request per problem, no early stop),
so the comparison is apples-to-apples on one scheduler. Continuous runs first (cold
prefix cache); naive second reuses the warm cache — a conservative bias toward naive.

    modal run evals/modal_verifier_sched.py
    modal run evals/modal_verifier_sched.py --n-problems 60 --b 4
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

app = modal.App("dexa-verifier-sched")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip("\n")


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, N: int, B: int, n_problems: int) -> None:
    import inspect
    import subprocess
    import sys
    from time import perf_counter

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams

    he = load_dataset("openai/openai_humaneval", split="test")
    if n_problems > 0:
        he = he.select(range(min(n_problems, len(he))))
    n = len(he)
    probs = list(zip(he["prompt"], he["test"], he["entry_point"]))

    tok = AutoTokenizer.from_pretrained(model)
    token_prompts = []
    for p in he["prompt"]:
        msg = [{"role": "user", "content":
                f"Complete this Python function. Return the full function in a "
                f"```python block.\n\n```python\n{p}\n```"}]
        ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        token_prompts.append({"prompt_token_ids": ids})

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=model, max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=True))
    print(f"[api] add_request{inspect.signature(engine.add_request)}", flush=True)

    def sp(nn, seed):
        return SamplingParams(n=nn, temperature=0.8, top_p=0.95, max_tokens=640, seed=seed)

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

    # ---- CONTINUOUS SCHEDULER (no global rounds, abort-on-pass) -------------
    solved = [False] * n
    launched = [0] * n          # samples issued for this problem
    inflight = [0] * n          # samples currently decoding for this problem
    active: dict[str, int] = {}  # request_id -> problem idx
    sched_tokens = 0

    def launch(idx: int, k: int) -> None:
        nonlocal sched_tokens
        for _ in range(k):
            rid = f"c{idx}:{launched[idx]}"
            engine.add_request(rid, token_prompts[idx], sp(1, launched[idx] + 1))
            active[rid] = idx
            launched[idx] += 1
            inflight[idx] += 1

    t0 = perf_counter()
    for idx in range(n):
        launch(idx, min(B, N))
    while engine.has_unfinished_requests():
        for out in engine.step():
            if not out.finished:
                continue
            idx = active.pop(out.request_id, None)
            if idx is None:
                continue                      # already aborted sibling
            inflight[idx] -= 1
            sched_tokens += len(out.outputs[0].token_ids)
            if solved[idx]:
                continue
            if passes(out.outputs[0].text, probs[idx]):
                solved[idx] = True
                for rid, j in [(r, j) for r, j in active.items() if j == idx]:
                    engine.abort_request(rid)
                    active.pop(rid, None)
                    inflight[idx] -= 1
            elif inflight[idx] == 0 and launched[idx] < N:
                launch(idx, min(B, N - launched[idx]))
    sched_wall = perf_counter() - t0
    sched_pass = sum(solved) / n

    # ---- NAIVE baseline on the SAME engine (n=N per problem, no early stop) -
    naive_active: dict[str, int] = {}
    naive_pass_flag = [False] * n
    naive_tokens = 0
    t0 = perf_counter()
    for idx in range(n):
        engine.add_request(f"n{idx}", token_prompts[idx], sp(N, 0))
        naive_active[f"n{idx}"] = idx
    while engine.has_unfinished_requests():
        for out in engine.step():
            if not out.finished:
                continue
            idx = naive_active.pop(out.request_id, None)
            if idx is None:
                continue
            naive_tokens += sum(len(o.token_ids) for o in out.outputs)
            naive_pass_flag[idx] = any(passes(o.text, probs[idx]) for o in out.outputs)
    naive_wall = perf_counter() - t0
    naive_pass = sum(naive_pass_flag) / n

    tok_ratio = naive_tokens / sched_tokens if sched_tokens else 0.0
    wall_ratio = naive_wall / sched_wall if sched_wall else 0.0
    avg_samples = sum(launched) / n

    print("\n" + "=" * 76)
    print(f"CONTINUOUS-BATCH VERIFIER SCHEDULER — HumanEval, {model} ({GPU})")
    print(f"N={N}, B={B}, {n} problems")
    print("=" * 76)
    print(f"  {'metric':18} {'naive n=N':>14} {'continuous sched':>18} {'ratio':>8}")
    print(f"  {'pass@'+str(N):18} {naive_pass:>14.3f} {sched_pass:>18.3f}")
    print(f"  {'decode tokens':18} {naive_tokens:>14} {sched_tokens:>18} {tok_ratio:>7.2f}x")
    print(f"  {'END-TO-END wall(s)':18} {naive_wall:>14.1f} {sched_wall:>18.1f} {wall_ratio:>7.2f}x")
    print(f"  avg samples/problem (sched): {avg_samples:.2f} / {N}")
    print("=" * 76)
    print(f"  HEADLINE: continuous scheduler matches naive pass@{N} "
          f"({sched_pass:.3f} vs {naive_pass:.3f}) at {tok_ratio:.1f}x fewer tokens AND "
          f"{wall_ratio:.1f}x less END-TO-END wall time — verify+abort inside one live batch.")
    print("=" * 76)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", n: int = 16, b: int = 4,
         n_problems: int = 0) -> None:
    print(f"continuous-batch verifier scheduler on {GPU}: {model}, N={n}, B={b}")
    run.remote(model, n, b, n_problems)
