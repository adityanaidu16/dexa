"""Where does test-time compute (sampling / reasoning-search) buy quality per dollar?

An opportunity-discovery eval: maps accuracy vs output-tokens across N=1/4/16 samples
on three tasks spanning the spectrum —

  * GSM8K (math)      — reasoning-bound; self-consistency (majority vote) should help
  * HumanEval (code)  — reasoning-bound; best-of-N (any sample passes tests) should help
  * TriviaQA (facts)  — knowledge/retrieval-bound; sampling should be ~FLAT (the model
                        either knows it or not) -> RETRIEVAL, not search, is the lever

The shape of each curve locates the value: a STEEP accuracy-vs-tokens frontier means
test-time reasoning-search is worth building for that task type; a FLAT one means the
answer isn't derivable by thinking harder and retrieval is what's needed. This is the
"quality-per-dollar map" that decides reasoning-search vs RAG vs interleave.

vLLM shares the prompt KV across the N samples natively (SamplingParams(n=N)), so the
only cost that grows is decode tokens — which is exactly what we bill against.

Verifiers are pure/free: exact numeric match (GSM8K), unit-test execution (HumanEval),
alias match (TriviaQA). Pure helpers are unit-tested in test_eval_verifiers.py.

    modal run evals/modal_eval_compute.py
    modal run evals/modal_eval_compute.py --model Qwen/Qwen2.5-7B-Instruct --n-problems 200
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Optional

import modal

GPU = os.environ.get("DEXA_EVAL_GPU", "A100-80GB")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.24.0", "datasets", "numpy")
    # hf-xet's CDN 403s here; removing it makes huggingface_hub fall back to plain
    # HTTP downloads (HF_HUB_DISABLE_XET alone wasn't honored).
    .run_commands("python -m pip uninstall -y hf-xet || true")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

app = modal.App("dexa-eval-compute")
hf_cache = modal.Volume.from_name("dexa-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Pure verifiers / extractors (no GPU; unit-tested in test_eval_verifiers.py).
# ---------------------------------------------------------------------------
def extract_final_number(text: str) -> Optional[float]:
    """Last numeric value in a solution (handles '#### 42', '$1,200', '-3.5')."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    n = nums[-1].replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(n)
    except ValueError:
        return None


def majority(xs: list) -> Optional[object]:
    """Most common non-None value (self-consistency vote)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return Counter(xs).most_common(1)[0][0]


def extract_python(text: str, entry_point: str) -> str:
    """Pull the code out of a chat completion: prefer the last ```python block, else
    the raw text. If it doesn't define the target function, it's a bare body."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip("\n")


def normalize_answer(s: str) -> str:
    """Lowercase, drop punctuation/articles/extra space (SQuAD/TriviaQA style).

    Punctuation is removed to *nothing* (not a space) so 'U.S.A.' -> 'usa' rather than
    exposing a standalone 'a' that the article filter would then eat."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def trivia_match(prediction: str, aliases: list[str]) -> bool:
    """True if any normalized gold alias appears in the normalized prediction."""
    p = normalize_answer(prediction)
    return any(normalize_answer(a) and normalize_answer(a) in p for a in aliases)


# ---------------------------------------------------------------------------
# GPU eval.
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/cache/hf": hf_cache})
def run(model: str, n_list: list[int], n_problems: int) -> None:
    import subprocess
    import sys

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.9,
              enforce_eager=True)

    def generate(prompts: list[str], N: int, max_tokens: int):
        sp = SamplingParams(n=N, temperature=(0.0 if N == 1 else 0.8), top_p=0.95,
                            max_tokens=max_tokens, seed=0)
        convos = [[{"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(convos, sp, use_tqdm=False)
        texts = [[o.text for o in r.outputs] for r in outs]
        toks = sum(len(o.token_ids) for r in outs for o in r.outputs)
        return texts, toks

    results: dict[str, list] = {}

    # ---- GSM8K: self-consistency (majority vote) --------------------------
    gsm = load_dataset("openai/gsm8k", "main", split=f"test[:{n_problems}]")
    gsm_prompts = [f"Solve step by step. End with '#### <number>'.\n\nProblem: {q}"
                   for q in gsm["question"]]
    gsm_gold = [extract_final_number(a) for a in gsm["answer"]]
    for N in n_list:
        texts, toks = generate(gsm_prompts, N, 512)
        correct = 0
        for samples, g in zip(texts, gsm_gold):
            pred = majority([extract_final_number(s) for s in samples])
            if pred is not None and g is not None and abs(pred - g) < 1e-4:
                correct += 1
        results.setdefault("gsm8k", []).append((N, correct / len(gsm_gold), toks))
        print(f"[gsm8k] N={N:2d} acc={correct/len(gsm_gold):.3f} tokens={toks}", flush=True)

    # ---- HumanEval: best-of-N (any sample passes the tests) ---------------
    he = load_dataset("openai/openai_humaneval", split="test")
    he_prompts = [f"Complete this Python function. Return the full function in a "
                  f"```python block.\n\n```python\n{p}\n```" for p in he["prompt"]]
    for N in n_list:
        texts, toks = generate(he_prompts, N, 640)
        passed = 0
        for samples, prob in zip(texts, zip(he["prompt"], he["test"], he["entry_point"])):
            prompt_code, test_code, entry = prob
            ok = False
            for s in samples:
                code = extract_python(s, entry)
                if f"def {entry}" not in code:
                    code = prompt_code + code
                program = f"{code}\n{test_code}\ncheck({entry})\n"
                try:
                    r = subprocess.run([sys.executable, "-c", program], timeout=10,
                                       capture_output=True)
                    if r.returncode == 0:
                        ok = True
                        break
                except Exception:
                    pass
            passed += int(ok)
        results.setdefault("humaneval", []).append((N, passed / len(he), toks))
        print(f"[humaneval] N={N:2d} pass={passed/len(he):.3f} tokens={toks}", flush=True)

    # ---- TriviaQA: self-consistency on a knowledge task (expect flat) ------
    tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                      split=f"validation[:{n_problems}]")
    tq_prompts = [f"Answer with just the answer, no explanation.\n\nQuestion: {q}"
                  for q in tq["question"]]
    tq_aliases = [a["aliases"] + a.get("normalized_aliases", []) for a in tq["answer"]]
    for N in n_list:
        texts, toks = generate(tq_prompts, N, 48)
        correct = 0
        for samples, aliases in zip(texts, tq_aliases):
            # self-consistency: most common normalized short answer, then match
            norm = majority([normalize_answer(s.strip().splitlines()[-1] if s.strip() else "")
                             for s in samples])
            if norm and trivia_match(norm, aliases):
                correct += 1
        results.setdefault("triviaqa", []).append((N, correct / len(tq_aliases), toks))
        print(f"[triviaqa] N={N:2d} acc={correct/len(tq_aliases):.3f} tokens={toks}", flush=True)

    # ---- frontier report --------------------------------------------------
    print("\n" + "=" * 74)
    print(f"QUALITY-PER-DOLLAR MAP — {model} ({GPU}, {n_problems} problems)")
    print("=" * 74)
    for task, rows in results.items():
        base_acc, base_tok = rows[0][1], rows[0][2]
        top_acc, top_tok = rows[-1][1], rows[-1][2]
        d_acc = top_acc - base_acc
        d_tok = top_tok - base_tok
        steep = (d_acc / (d_tok / 1e6)) if d_tok > 0 else 0.0  # acc gain per 1M extra tok
        verdict = "STEEP -> reasoning-search pays" if d_acc >= 0.05 else "FLAT -> retrieval is the lever"
        print(f"\n{task}:")
        for N, acc, toks in rows:
            print(f"   N={N:2d}  acc={acc:.3f}  tokens={toks:>9}")
        print(f"   N1->N{rows[-1][0]}: +{d_acc:.3f} acc for {d_tok/1e6:.2f}M extra tokens "
              f"({steep:+.2f} acc/1M) -> {verdict}")
    print("=" * 74)
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "unsloth/Llama-3.1-8B-Instruct", n_problems: int = 150,
         samples: str = "1,4,16") -> None:
    n_list = [int(x) for x in samples.split(",")]
    print(f"compute-allocation eval on {GPU}: {model}, N={n_list}, {n_problems} problems")
    run.remote(model, n_list, n_problems)
