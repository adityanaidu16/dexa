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

# KV cache interchangeability — across FORMATS (yes) and across WEIGHTS (no)

**Run:** `modal run evals/modal_kv_interchange.py`, Llama-3.1-8B Instruct + base, single
diverse 256-token context (no repetition), greedy 48-token continuation, agreement vs
each model's own reference. Plain HF/torch for exact KV control.

*(Methodology note: this took three iterations to measure honestly. A tiled prompt made
greedy decode an in-context COPY task (induction heads) that is robust to any KV — every
format AND cross-model showed a false 100%. Only a single-pass, non-repeated context
forces genuine generation and exposes the real differences below. The first "cross-model
KV is lossless" reading was an artifact; do not trust interchange tests on repeated text.)*

## FORMAT axis — same weights, KV round-tripped through a numeric format

| format | bytes/tok | token agreement (48 greedy) | first divergence | step-1 KL |
|--------|----------:|----------------------------:|-----------------:|----------:|
| fp16 | 1.0× | 100% | — | 0 |
| **fp8 (e4m3)** | **0.5×** | **100%** | — | 0.0003 |
| **int8** | **0.5×** | **100%** | — | 0 |
| int4 | 0.25× | 29% | token 14 | 0.0027 |

**KV is a portable container down to int8/fp8 — 2× compression, byte-identical greedy
generation over 48 tokens.** int4 (per-(token,head) symmetric) breaks (diverges at token
14), so 4× needs a smarter scheme (group-wise/asymmetric, or keep a few outlier channels
in higher precision). This is the interchangeability that *works*: store/transport the KV
in fp8/int8 and any deployment of the **same weights** loads it losslessly — halves what a
persistence/offload layer must move, and enables mixed-precision serving.

## WEIGHTS axis — inject one model's KV into another (base ↔ instruct, same 8B arch)

| direction | KV cosine (K / V) | token agreement | first divergence |
|-----------|------------------:|----------------:|-----------------:|
| base gen ← instruct KV | 0.984 / 0.945 | 12.5% | token 5 |
| instruct gen ← base KV | 0.984 / 0.945 | 4.2% | token 2 |

**Raw KV is NOT interchangeable across weights — not even a light fine-tune.** base and
instruct are the *same architecture* and their KV is numerically very close (cosine 0.98
K / 0.95 V), yet generation diverges within **2–5 tokens** into different (coherent but
distinct) continuations. The lesson: **high KV cosine does not imply interchangeability** —
small per-layer differences compound through the autoregressive loop and flip the argmax
almost immediately. A KV store must be keyed to the exact model weights; you cannot share
a cache between two fine-tunes and expect the same output.

**Verdict.** Formats: interchangeable (fp8/int8 free, int4 needs work). Weights: not —
KV is model-specific. *One promising thread the numbers hint at:* the cross-fine-tune KV
is close (0.98 cosine), so a cheap learned "KV bridge" (base-KV → instruct-KV transform)
might recover reusability across fine-tunes of a shared base — the one place the weights
axis isn't hopeless, and a genuinely novel direction if worth pursuing.

*(Caveats: greedy decode, one model family, one 256-token passage, 48-token horizon,
per-(token,head) symmetric quant for int8/int4. Directional, not a sweep.)*

## The SGLang gate — is the tree-native decode white space real? (No.)

**Run:** `modal run evals/modal_sglang_pathology.py`, Llama-3.1-8B, A100-80GB. The vLLM
run showed 8-wide decode over a long shared prefix costs 174× a single decode at 128k
(the shared prefix KV re-read per branch). This tests whether SGLang's RadixAttention —
the closest existing thing to a tree-native stack — has the same weakness. Measured
decode/step by differencing two generation lengths on an already-warm shared prefix
(cancels prefill + cache effects; the naive full-minus-prefill was polluted by
RadixAttention caching the prefix across calls). Branches get a unique final token so
they're distinct sequences sharing the prefix, not deduped.

| context | decode/step n=1 | n=8 | n=8 vs n=1 | vLLM was |
|--------:|----------------:|----:|-----------:|---------:|
| 32k | 20.9 ms | 21.7 ms | **1.04×** | ~19× |
| 128k | 21.1 ms | 46.7 ms | **2.2×** | **174×** |

**SGLang handles it — flat in branch count to 32k, sublinear (2.2× for 8 branches) at
128k, versus vLLM's 174× blowup.** RadixAttention reads the shared prefix once and
batches the branch queries against it. So vLLM's pathology is vLLM-specific, not an
industry gap. (Caveat: the 4k n=1 point landed in timing noise → its ratio is a
divide-by-~0 artifact; the absolute per-step numbers are the clean signal.)

**Strategic consequence.** The one architectural white space our experiments pointed to —
an efficient shared-prefix/tree-native branched decode — is already occupied by SGLang.
Building a new inference stack for it would be reinventing SGLang against a mature
open-source incumbent. Combined with the rest of the ledger, every path *inside* the
inference stack is now closed for a small team: orchestration is commoditized (Temporal/
a loop), the kernel/scheduler layer is owned and well-executed by SGLang, and KV
persistence/interchange is commoditized or model-specific. The durable opportunity, if
any, is *up* the stack — a vertical application that owns a workflow and its eval/quality
data (code-gen quality being the one large, durable lever the evals found) and *uses*
SGLang rather than competing with it. The inference tricks are inputs, not the product.

## Agentic-serving value benchmark v1 — and why the naive story mostly reduces to prefix caching

**Run:** `modal run evals/modal_agentic_value.py`, Llama-3.1-8B on SGLang, A100-80GB.
Per agent turn (shared context W, B-way branch, verify, early-stop), GPU-seconds
(dedicated-GPU wall-clock) under three conditions, swept over W and B.

| W | B | reprefill (no cache) | cached | agentic | vs reprefill | vs cached | decode tok |
|--:|--:|---------------------:|-------:|--------:|-------------:|----------:|-----------:|
| 4k | 4 | 3.68s | 2.66s | 2.66s | 1.4× | 1.00× | 2× fewer |
| 16k | 8 | 13.27s | 2.68s | 2.66s | 5.0× | 1.01× | 4× fewer |
| 32k | 8 | 30.65s | 2.70s | 2.66s | **11.5×** | **1.02×** | 4× fewer |

**The result partly falsifies the naive value story:**
1. The large, W×B-scaling win (up to 11.5×) is entirely **prefix caching** (RadixAttention),
   which is table stakes — free in SGLang, standard in every serious provider. Not a wedge.
2. Against a realistic caching baseline, agentic wall-clock is **~1.0×** — no GPU-time win.
   SGLang batches branches so efficiently that generating 8 costs the same wall-clock as 2,
   so early-stop saves **decode tokens (2–4×) but not GPU-seconds on a dedicated GPU**.
3. Early-stop runs branches in sequential waves, so it can *increase* latency on harder
   turns (more waves) — a cost/latency tradeoff, not a free win.

**Where the value actually is, by how the buyer pays:**
- **Token-billed (generic API):** early-stop = 2–4× fewer decode tokens on the branching =
  a direct, defensible bill reduction today.
- **GPU-time (self-hosting):** the early-stop win only appears under serving **contention**
  (many concurrent agents saturating the GPU: fewer tokens → more agents/GPU). The
  dedicated-GPU wall-clock here can't show it — the benchmark's blind spot.

**The flaw and the next test.** A single idle GPU underutilized by one agent's small batch
hides early-stop's value. The benchmark that would actually prove (or kill) the wedge must
model **throughput under contention**: agents-per-GPU-per-second at a fixed latency SLA,
generic serving vs early-stop serving, as concurrency rises. That is the real cost axis for
a self-hosting buyer, and it's the honest v2.

## Contention v2 — early-stop under load: real but modest, and not a moat

**Run:** `modal run evals/modal_contention.py`, Llama-3.1-8B on SGLang, A100-80GB, W=4k,
B=8, early-stop k=2. Sweeps concurrency C of agents (each with its own W-context, distinct
per agent → no cross-agent sharing) and measures agent-turns/sec: generic (all B) vs
agentic (early-stop k).

| concurrency | generic turns/s | agentic turns/s | ratio |
|------------:|----------------:|----------------:|------:|
| 1  | 0.08 | 0.33 | 4.0× (cold-start artifact — ignore) |
| 8  | 0.86 | 1.18 | 1.38× |
| 32 | 0.93 | 1.57 | 1.69× |
| 96 | 0.88 | 1.65 | **1.87×** |

(The C=1 "4.0×" is a JIT/cold-start spike on the first timed batch; the script's auto-verdict
wrongly compared it to C=96. The real trend, C=8→96, GROWS 1.38→1.87 as both policies saturate.)

**Findings:**
1. Early-stop **does** pay under contention — steady-state **~1.85× throughput** (generic caps
   ~0.9 turns/s, agentic ~1.65). This is the win v1's idle GPU couldn't show. The self-hosting
   cost story is real, not dead.
2. But it's ~1.85×, **not** the 4× that the token ratio (B/k) predicts, because LLM decode is
   memory-bandwidth-bound: the per-step weight load is amortized across all sequences in the
   batch, so extra branches are partly free even under load. Fewer branches saves less than
   token-counting implies.
3. Still not a moat: a buyer captures most of the 1.85× with round-based early-stop in their
   **own** orchestration (send k, check, send more) on any provider. Doing it in-engine is
   tighter but not defensibility.

**Verdict for the platform thesis.** The best case for the inference-stack cost wedge is
~1.85× throughput under load, replicable by the buyer without the platform. A real efficiency,
not a business. Combined with the rest of the ledger, every inference-stack value story tested
here has resolved to table stakes (prefix caching), engine-owned (SGLang branched decode), or
buyer-replicable orchestration (early-stop). The durable opportunity is up-stack — a vertical
app owning a workflow and its eval/quality data — not in the serving layer.

# Multimodal execution thesis — the first prize that survived measurement

**Run:** `modal run evals/modal_vlm_frontier.py`, Qwen2.5-VL-7B on vLLM, A100-80GB,
DocVQA (200 high-res document images, relaxed-match accuracy). Sweeps the visual-token
budget via input resolution and measures accuracy vs throughput. (v1 was invalid: ChartQA
images too small for visual tokens to dominate, and the first budget ate cold-start; v2
uses high-res docs + a warmup batch + throughput as the metric.)

| budget | accuracy | img/s | visual tokens | throughput | Δacc |
|-------:|---------:|------:|--------------:|-----------:|-----:|
| 1536px | 0.935 | 11.4 | 2201 | 1.0× | — |
| **1024px** | **0.925** | 30.2 | 1017 | **2.7×** | **−1.0%** |
| 768px | 0.885 | 47.2 | 574 | 4.1× | −5.0% |
| 512px | 0.735 | 78.6 | 277 | 6.9× | −20% |
| 384px | 0.445 | 98.3 | 182 | 8.6× | −49% |

**Cut cost ~2.7× at ~1% accuracy, ~4× at ~5%** — on high-res docs where visual tokens
(2201) dominate the LLM's work. This is the first thesis in the whole ledger whose prize
did NOT evaporate under rigorous measurement. The degradation is sensible (below ~768px
small text becomes unreadable), confirming the eval is real.

**Honest caveat — prize vs moat.** The lever here is naive resolution reduction, which a
client can do themselves (resize before any API). So this run *sizes the prize* but isn't
yet the moat. The execution-owned, non-commoditized edge is (1) content-aware
in-forward-pass token pruning (FastV-style) that a gateway can't touch and should BEAT the
resize frontier by keeping informative tokens, and (2) adaptive per-request budgeting.

**Next test (the moat test):** does content-aware in-model pruning beat this resize
frontier — same accuracy at higher throughput? If yes, execution-owned multimodal token
reduction is a real, defensible gain no gateway can replicate. Beachhead = high-res
document AI (invoices, forms, reports, contracts) where visual tokens dominate cost.
