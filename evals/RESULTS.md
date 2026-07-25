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

## Live engine — the post-hoc number, measured end-to-end

**Run:** `modal run evals/modal_verifier_engine.py`, same model/GPU, N=16, B=4, all 164
problems. Not reconstructed: the engine actually generates in batched rounds of 4, runs
the unit-test verifier after each round, drops solved problems, and only survivors
continue. Both wall-clock and tokens are measured live.

| metric | naive n=16 | live engine (B=4 rounds) | ratio |
|--------|-----------:|-------------------------:|------:|
| pass@16 | 0.896 | 0.878 | ~equal (i.i.d. redraw noise) |
| decode tokens | 858,658 | 330,174 | **2.60×** |
| GPU generation wall (s) | 168.1 | 110.0 | **1.53×** |
| verify wall (s) | 27.0 | 18.6 | — |

The live token saving (**2.60×**) confirms the post-hoc 2.85× — the small gap is exactly
the "pay for all B in the passing round" overhead the model predicted. Quality holds
(0.878 vs 0.896 = ~3 problems, sampling noise from fresh per-round draws). Wall-clock
improves less than tokens (**1.53×**) because the batch *shrinks* each round — 27 hard
problems left by round 2 underfill the A100, so late rounds are latency-bound, not
throughput-bound. That gap is the real engineering surface: cross-problem round
pipelining and continuous batching would recover most of the token→wall-clock slack.
Even naively, though, the engine is **2.6× cheaper and 1.5× faster at equal quality —
live, not on paper.**

## Continuous-batch scheduler — the wall-clock gap, closed

**Run:** `modal run evals/modal_verifier_sched.py`, same model/GPU, N=16, B=4, all 164
problems. The round engine left wall-clock on the table because discrete global rounds
drain the batch (only the hard tail is left in round 4, idling the A100). This drives
vLLM's low-level `LLMEngine` step loop directly (`add_request`/`step`/`abort`) with **no
rounds**: all problems' samples share one continuously-batched flight, finished samples
are verified each step, a problem's siblings are **aborted the instant one passes**, and
a failed wave's next samples drop straight into the same live batch. Baseline is the
same engine run with no early stop (N independent n=1 requests/problem).

| metric | naive (no early stop) | continuous scheduler | ratio |
|--------|----------------------:|---------------------:|------:|
| pass@16 | 0.884 | 0.884 | **identical** |
| decode tokens (incl. aborted partials) | 862,050 | 327,607 | **2.63×** |
| **END-TO-END wall (s)** | 183.8 | 86.6 | **2.12×** |

**pass@16 is *exactly* equal (0.884 = 0.884)** — same seeds draw the same 16 samples per
problem, and early-stop only skips samples *after* a pass, so it cannot change whether
any of the 16 would have passed. Early-stop is provably lossless here, not approximately.

**The gap is closed: 1.5× → 2.12× wall-clock**, now tracking the 2.63× token saving
closely (the residual is verify latency on the critical path + per-step overhead). This
is the step from benchmark to engine: **2.1× faster end-to-end AND 2.6× cheaper at
byte-identical quality**, because verification and abort live *inside* one continuously
batched flight instead of between synchronized rounds. This adaptive per-sequence control
(verify → abort → refill, all mid-batch) is exactly what vLLM's static `n=16` cannot do.

## Robustness — 2nd model + a weak, deployable verifier

**Run:** `modal run evals/modal_robustness.py`, N=16, all 164 problems. Two fair
objections to the headline: it used one model and an *oracle* verifier (the full hidden
suite). Real deployments see only a few **visible** example tests — which can
false-positive (pass visible, fail hidden) or miss (a hidden-correct sample never trips
the truncated check). So we score every sample on both the full hidden suite (ground
truth + oracle signal) and a weak verifier = **first 3 asserts only**; the weak engine
ships the first visible-pass sample and is graded on hidden — exactly what production does.

| model | oracle pass@16 | weak pass@16 | % ceiling kept | visible FP rate | oracle cost | weak cost |
|-------|---------------:|-------------:|---------------:|----------------:|------------:|----------:|
| Llama-3.1-8B-Instruct | 0.909 | 0.848 | **93%** | 7.7% | 2.85× | **3.04×** |
| Qwen2.5-Coder-7B-Instruct | 0.963 | 0.921 | **96%** | 3.8% | 3.45× | **3.52×** |

**The win holds on both axes.** A realistic 3-example verifier keeps **93–96%** of the
oracle pass@16 while cutting cost **3.0–3.5×** — actually *cheaper* than the oracle
verifier, because 3 asserts trip sooner than the full suite. The second model (a coding
model) doesn't just replicate the shape, it strengthens it: higher ceiling (0.963),
more quality kept (96%), lower false-positive rate (3.8%), bigger speedup (3.5×).

**The honest cost of a weak verifier, quantified:** it's not free. ~4–8% of the ceiling
is lost to visible false-positives (a sample that passes 3 asserts but fails hidden gets
shipped) plus the 6–10 problems per model where no sample passes visible within N. A
stronger/better-calibrated model loses less. That is the real tradeoff to state to a
buyer — not "lossless," but "keep >90% of the search gain at 3× less cost with only the
tests you already have." (Caveat: 'visible = first 3 asserts' is a proxy for real example
tests; HumanEval doesn't ship a clean visible/hidden split.)

## Product implication

The eval-driven answer to "where do we add value": **an efficient, verifier-guided
search engine for verifiable reasoning — code first.** Beachhead = code generation /
coding agents, where (a) the quality lever is largest (+21%), (b) verifiers are free
(unit tests), and (c) the naive 16× cost is the inefficiency to capture — and we now
have the hard number for that capture (**2.8–4.7× cheaper at equal quality**). Not
knowledge QA (retrieval's job). The MVP and the proof are the same artifact: a
verifier-guided early-stop engine that delivers pass@16 quality at pass@4 cost — and,
with the continuous-batch scheduler, at **2.1× the throughput** of naive best-of-N too.

*(Efficiency caveats: batched B=4 shares the prompt KV across a round but pays a small
tail-latency cost vs B=1; the 2.85× is the conservative, realistic figure. The verifier
here is the full hidden test suite — a real deployment sees only visible tests, which
recovers slightly less of the gain but keeps the same shape. pass@k is an oracle
selector, so 0.909 is the sampling ceiling a real verifier approximates.)*

*(Caveats: single model (Llama-3.1-8B); HumanEval "best-of-N" is pass@k — an oracle
selector — so it's the quality *ceiling* of sampling, which a real verifier
approximates; TriviaQA self-consistency has verifier noise but the flat shape is
robust. Directional, not a leaderboard.)*

---

# Long-context branching curve — the thesis test that came back inverted (and why that's useful)

**Run:** `modal run evals/modal_context_scaling.py`, Llama-3.1-8B, A100-80GB, N=8
branches, gen=128, prefix caching OFF (models the at-scale regime where KV can't stay
resident). Compares **shared** (one request, `n=N` — prefill the context once, branch N)
vs **naive** (N sequential requests, each re-prefills the full context) across context
length L.

| context L | prefill s | shared s (n=N) | naive s (N re-prefill) | speedup |
|----------:|----------:|---------------:|-----------------------:|--------:|
| 4k   | 0.27  | 7.52   | 22.77  | **3.03×** |
| 16k  | 1.35  | 13.82  | 32.16  | 2.33× |
| 32k  | 3.48  | 32.57  | 48.90  | 1.50× |
| 64k  | 10.06 | 91.76  | 101.41 | 1.11× |
| 128k | 32.68 | 285.72 | 283.16 | **0.99×** |

**The branch-sharing speedup SHRINKS with context (3.0× → 1.0×) — the opposite of the
naive thesis.** But the decomposition shows *why*, and it's the actually-useful result:

- **The prefill prize is real and grows.** At 128k, one prefill is 32.7 s and dominates
  per-branch decode (~2.7 s). Paying prefill once vs 8× *should* be a ~5× win.
- **vLLM's native `n=N` FAILS to capture it at long context.** Its concurrent branched
  decode over the long shared KV is pathological: per-step 8-wide decode is **2.8× a
  single decode at 4k (sublinear, good) but 94× at 128k (massively superlinear).** So
  the shared run spends ~253 s in decode and ends up *no faster than re-prefilling
  everything* (285 s ≈ 283 s). The prefill saving is real but eaten by a decode-side
  inefficiency — it is not reading the shared prefix KV once and reusing it across the 8
  branches; it re-pays per branch.
- **The memory wall confirms the regime.** You *cannot* run 8 independent 128k contexts
  concurrently (8×~16 GB = 128 GB > 80 GB HBM), so naive best-of-N at long context must
  serialize. Sharing the prefix is the only thing that fits — but you also need an
  efficient branched decode, which vLLM does not provide.

**The redirect.** The bottleneck at long context is **decode over the long KV, not
prefill** — exactly the decode-attention tax flagged earlier. The unrealized prize
(≈5× at 128k: prefill-once + a decode that reads the shared prefix once/step ≈ 54 s vs
vLLM's 285 s) points the product away from "persist/reuse KV" (commoditized, and vLLM
already shares prefill *within* a request) and toward **an efficient shared-prefix
branched decode** (tree/shared-prefix attention that reads the long prefix KV once and
applies all N branch queries) and/or **KV compression** to cut the decode cost directly.

*(Caveats: one run, `enforce_eager` (no CUDA graphs — but eager penalizes the
many-small-steps naive baseline MORE than shared, so removing it would make shared look
*worse*, not better; the shrink is robust to that). The ~5× "ideal" is inferred from the
timing decomposition, not yet directly measured — a shared-prefix/tree-attention decode
kernel is needed to realize it. The n=N long-context decode blowup should be reproduced
across an n∈{1,2,4,8} sweep before it's called a definitive vLLM pathology.)*

## Decode-pathology sweep — the blowup, isolated and confirmed

**Run:** `modal run evals/modal_decode_pathology.py`, Llama-3.1-8B, A100-80GB, gen=64,
prefix caching off, eager. Isolates per-step decode cost across context length L and
branch count n. `decode/step = (t_full − t_prefill)/gen`; `ratio = per_step(n)/per_step(1)`.

| context L | decode/step n=1 | n=2 | n=4 | n=8 | ratio @ n=8 (ideal 8×) |
|----------:|----------------:|----:|----:|----:|-----------------------:|
| 4k   | 21.7 ms | 26.4 ms | 34.5 ms | 100.9 ms | **4.6×** (healthy, sublinear) |
| 32k  | 21.9 ms | 76.7 ms | 188 ms | 423 ms | **19.3×** |
| 128k | 21.9 ms | **554 ms** | 1643 ms | 3812 ms | **174×** |

**Two facts nail it:**
1. **Single-branch decode is flat across context — ~22 ms/step at 4k, 32k, AND 128k.**
   One sequence reading even a 16 GB / 128k KV per step is cheap and well-optimized. The
   KV *size* is not the problem.
2. **The moment n>1 at long context, per-step cost explodes.** At 128k, going from n=1 to
   n=2 jumps 22 ms → 554 ms (**25×** for *one* extra branch), reaching **174× at n=8**
   against an ideal of 8×. At 4k the same n=8 is only 4.6× (sublinear, correct). So the
   blowup is specific to **parallel sampling over a long shared prefix**: vLLM is not
   reusing the shared prefix KV across branches during decode — it re-pays per branch
   (copy-on-write fork or a non-shared attention path). Eager mode *compresses* this
   ratio, so the true blowup is even worse.

**The opportunity, quantified.** An efficient shared-prefix branched decode — read the
long prefix KV **once** per step, batch the N branch queries against it (tree /
shared-prefix attention) — should cost roughly n=1 plus a little. At 128k that's ~tens of
ms vs vLLM's 3812 ms: a **~20–95× decode speedup** for long-context branching (best-of-N,
parallel tool-calls, reasoning trees over a big shared context). That is the concrete,
measured, differentiated wedge — and it's a *decode-kernel* problem, not a KV-offload
problem, so it sidesteps the commoditized LMCache/Mooncake lane entirely.

**The one check before committing (honest).** This is vLLM 0.24 *parallel sampling*
specifically. SGLang's RadixAttention is purpose-built to share prefix KV across
branches and **may already avoid this** — if so, the answer is "serve long-context
branching on SGLang," not "build a new kernel." The next experiment is the same sweep on
SGLang (and/or an explicit tree-attention path): if the blowup persists across engines,
there's a real gap to own; if SGLang is flat, the win is picking the right runtime. Do
NOT pitch the wedge until that's known.
