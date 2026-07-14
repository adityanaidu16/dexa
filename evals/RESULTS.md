# Compute-allocation eval — where does test-time search buy quality per dollar?

**Run:** `modal run evals/modal_eval_compute.py`, Llama-3.1-8B-Instruct, A100-80GB,
150 problems/task (HumanEval = all 164), N ∈ {1, 4, 16} samples.
GSM8K = self-consistency (majority vote); HumanEval = best-of-N (any sample passes the
unit tests); TriviaQA = self-consistency on a knowledge task.

| task | N=1 | N=4 | N=16 | Δacc (N1→16) | tokens N1→16 | verdict |
|------|----:|----:|-----:|-------------:|-------------:|---------|
| **HumanEval (code)** | 0.689 | 0.829 | **0.902** | **+21.3%** | 52k → 858k | **STEEP — search pays big** |
| GSM8K (math) | 0.880 | 0.887 | **0.933** | +5.3% | 33k → 540k | moderate |
| TriviaQA (facts) | 0.680 | 0.687 | 0.693 | +1.3% | 0.9k → 15k | **FLAT — retrieval, not search** |

## What it says (the value is located)

1. **Verifiable reasoning — especially CODE — is where test-time search pays.**
   Best-of-N with a unit-test verifier lifts HumanEval pass **0.69 → 0.90 (+21 pts)**.
   Math (self-consistency) is a real but smaller lever (+5 pts). These are exactly the
   tasks with **cheap, automatic verifiers** (tests, answer checkers).
2. **Knowledge/fact tasks are flat.** TriviaQA barely moves with sampling (+1.3 pt) —
   the model either knows the fact or it doesn't. Thinking harder can't add knowledge;
   **retrieval (RAG) is the lever there**, not search. A reasoning-search engine should
   *not* target knowledge QA.
3. **The cost is naive-16×, which is the opportunity.** The +21% on code cost ~16×
   tokens because the eval ran all N samples to completion. A real engine wouldn't:
   with a verifier you **stop as soon as one sample passes** and **prune failing
   branches early** — capturing most of the +21% at a fraction of the compute. That
   verifier-guided early-termination/pruning is precisely what vLLM does *not* provide
   (it gives `n>1` + shared-prefix KV, but no adaptive control), and it's the
   differentiated product.

## Product implication

The eval-driven answer to "where do we add value": **an efficient, verifier-guided
search engine for verifiable reasoning — code first.** Beachhead = code generation /
coding agents, where (a) the quality lever is largest (+21%), (b) verifiers are free
(unit tests), and (c) the naive 16× cost is the inefficiency to capture. Not knowledge
QA (retrieval's job). Next: prove the *efficiency* claim — verifier-guided
best-of-N with early-stop/pruning reaches ~pass@16 quality at a fraction of pass@16
cost, vs naive all-N.

*(Caveats: single model (Llama-3.1-8B); HumanEval "best-of-N" is pass@k — an oracle
selector — so it's the quality *ceiling* of sampling, which a real verifier
approximates; TriviaQA self-consistency has verifier noise but the flat shape is
robust. Directional, not a leaderboard.)*
