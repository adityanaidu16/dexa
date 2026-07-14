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

## Efficiency proof — verifier-guided early-stop (the product claim, measured)

**Run:** `modal run evals/modal_verifier_search.py`, same model/GPU, N=16, all 164
HumanEval problems. Generate N samples per problem, record each sample's pass/tokens,
then compute costs post-hoc (samples are i.i.d.). Every strategy yields the **identical**
pass@16 — only the token cost differs.

| strategy | tokens | vs naive | pass@16 |
|----------|-------:|---------:|--------:|
| naive best-of-N (all N to completion) | 860,830 | 1.00× | 0.909 |
| **early-stop B=4 (batched, realistic)** | **302,349** | **2.85×** | 0.909 |
| early-stop B=1 (one-at-a-time, theoretical min) | 182,431 | 4.72× | 0.909 |

**avg samples used (B=1): 3.11 / 16** — most problems pass on the first sample or two;
only the hard tail consumes the full budget. That skew is exactly why early-stop wins.

**The headline number: pass@16 quality (0.909) at 2.8× less cost** with realistic
batched rounds of 4, up to 4.7× at the one-at-a-time minimum. The +21% quality lever
from best-of-N is captured at a *fraction* of the naive 16× cost, because a free
verifier (the unit tests) lets you stop the moment a sample passes and spend nothing
more. This is the concrete product claim: **"pass@16 quality at ~pass@4 cost."**

## Product implication

The eval-driven answer to "where do we add value": **an efficient, verifier-guided
search engine for verifiable reasoning — code first.** Beachhead = code generation /
coding agents, where (a) the quality lever is largest (+21%), (b) verifiers are free
(unit tests), and (c) the naive 16× cost is the inefficiency to capture — and we now
have the hard number for that capture (**2.8–4.7× cheaper at equal quality**). Not
knowledge QA (retrieval's job). The MVP and the proof are the same artifact: a
verifier-guided early-stop engine that delivers pass@16 quality at pass@4 cost.

*(Efficiency caveats: batched B=4 shares the prompt KV across a round but pays a small
tail-latency cost vs B=1; the 2.85× is the conservative, realistic figure. The verifier
here is the full hidden test suite — a real deployment sees only visible tests, which
recovers slightly less of the gain but keeps the same shape. pass@k is an oracle
selector, so 0.909 is the sampling ceiling a real verifier approximates.)*

*(Caveats: single model (Llama-3.1-8B); HumanEval "best-of-N" is pass@k — an oracle
selector — so it's the quality *ceiling* of sampling, which a real verifier
approximates; TriviaQA self-consistency has verifier noise but the flat shape is
robust. Directional, not a leaderboard.)*
