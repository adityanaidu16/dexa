# Candidate E — Verifier-guided search inference for code ("pass@16 quality at pass@4 cost")

## 1. Thesis

A Morph-style specialized inference provider for coding agents. The API takes a prompt plus a cheap verifier (the customer's unit tests today), runs a continuous-batch verify-abort-refill scheduler inside the engine, and returns the first sample that passes. The customer sentence: "We send the prompt and the tests; we get back the first solution that passes and we pay for the solve, not for sixteen tries." The measured basis is narrow but real: on HumanEval with an 8B model, the scheduler matched naive best-of-16 pass rate exactly (0.884 = 0.884) at 2.63x fewer decode tokens and 2.12x less end-to-end wall-clock (01-evidence-ledger-dexa.md, verifier-early-stop). The ledger's verdict, which this brief accepts and designs around, is that the token saving is "buyer-replicable, not a moat" (docs/FINDINGS.md); the product case therefore rests on what a client-side loop cannot do (mid-batch abort, refill, per-solve pricing) and on unmeasured extensions (real repos, larger models) that section 6 prices as experiments.

## 2. Customer and workload

Who buys: teams shipping coding agents or code-generation features on open-weight models they already serve or rent. The research files show this buyer exists and is standardizing on open weights: OpenHands recommends Qwen3.6-35B-A3B on vLLM/SGLang with 32,768 context (05-research-agents.md, OpenHands docs); Cursor's Composer 2 started from the open-weight Kimi K2.5 base, in a partnership Moonshot says involved Fireworks (05-research-agents.md); Cognition serves SWE-1.6 and SWE-grep on Cerebras (05-research-agents.md); open-weight models carried a majority of OpenRouter tokens by mid-2026, "concentrated in coding and agentic workloads" (05-research-agents.md, Mozilla State of Open Source AI). Morph's founder-stated customers (create.xyz, databutton, continue.dev, Framer, Webflow, Block, Vercel; 05-research-morph.md) are the same buyer profile.

What they run today: an agent loop where the model is one step among many. TraceLab (4,300 Claude Code/Codex sessions) puts the median cached prefix per step at 116-126K tokens, ~860 new input tokens and ~200-250 output tokens, with a median human gap of 1.4 min (05-research-agents.md). Mooncake's Codex traces show ~131:1 input:output (05-research-agents.md). Search subagents consume 56.6% of tokens in coding traces per Relace and "60% of their turns" per Morph (05-research-morph.md). The candidate-E workload is the generation step at the end of that loop: one prompt (task + retrieved context), a handful of tests, and a decision to sample once or many times.

How they pay today: per token on generic APIs (Fireworks: prompt caching within one replica, 50% cached discount, session-affinity headers; 05-research-agents.md), per token on specialized providers (Morph Fast Apply $0.80/$1.20 per M tokens; Relace Apply 3 $0.85/$1.25; 05-research-morph.md), or per GPU-hour when self-hosting (Tensormesh reserved H200 $2.50/hr, 05-research-kvcache.md; MI355X rental $2.29-$2.95/GPU-hr, 05-research-wafer.md). All engine measurements here are Llama-3.1-8B-Instruct and Qwen2.5-Coder-7B-Instruct on A100-80GB, vLLM 0.24.0 (01-evidence-ledger-dexa.md); nothing at 30B-A3B or larger is measured.

## 3. The pain, in the customer's words

Real quotes from the research files, then our paraphrase (marked as such):

- "even claude is well over 11% error rates with search and replace" — Morph Launch HN (05-research-morph.md). The buyer already pays for a second model to fix a first model's mistakes.
- Delegating retrieval "save[s] on (valuable) agent tokens and avoid[s] polluting the agent's context" — Cognition SWE-grep (05-research-morph.md). Buyers accept specialized subcalls when they cut agent tokens.
- "a specialized model like FAS is only useful if you can actually separate search from the rest of the agentic coding task" — Relace (05-research-morph.md). The buyer's condition: the subcall must be cleanly separable. "Generate until the tests pass" is separable.
- A Codex user reported ~$15.9 for ~20M tokens across 112 model calls in 4 turns when caching failed (05-research-agents.md, openai/codex #25604). Token bills for agent loops are visible and resented.

Paraphrase (ours, not a quote): "Best-of-N with my tests gets me from 0.69 to 0.90 on this benchmark (01-evidence-ledger-dexa.md, compute-allocation-map), but I pay 16x tokens for it, so I don't do it in production." The repo's own words: "The cost is naive-16x, which is the opportunity" (evals/RESULTS.md, "What it says").

## 4. Value proposition and the proof-of-value benchmark

Metric: pass@N on a held-out test suite, decode tokens billed, and end-to-end wall-clock, all at fixed N and B. The claim is equal pass@N at fewer tokens and less wall-clock; never "higher pass rate".

Setup (the measured one, evals/modal_verifier_sched.py): all 164 HumanEval problems, N=16, B=4, temperature 0.8, top_p 0.95, max_tokens 640, seed 0, Llama-3.1-8B-Instruct, A100-80GB, vLLM 0.24.0, scheduler driving LLMEngine add_request/step/abort_request; verifier = the problem's check() run in a subprocess with a 10 s timeout (evals/modal_verifier_sched.py, passes()).

Baselines, named: (a) vanilla vLLM, same engine, N independent n=1 requests per problem with no early stop — measured (naive column); (b) round-based early stop, B=4 per llm.chat call with verify between rounds — measured (evals/modal_verifier_engine.py), and this is the proxy for what a buyer builds client-side on any provider; (c) SGLang under load — measured as the contention-v2 harness (evals/modal_contention.py); (d) LMCache/prefix-cached generic serving — measured in agentic-value v1 as the "cached" column. Fireworks/Together as external baselines: unmeasured; the experiment is in section 6.

Target numbers and what was measured: continuous scheduler 0.884 pass@16 vs naive 0.884; 327,607 vs 862,050 decode tokens (2.63x, aborted partials counted); 86.6 s vs 183.8 s end-to-end (2.12x) (evals/RESULTS.md, "Continuous-batch scheduler"). Round engine: 330,174 tokens (2.60x) but only 1.53x GPU generation wall (110.0 vs 168.1 s) because rounds drain the batch (evals/RESULTS.md, "Live engine"). Weak verifier (first 3 asserts of check(), scored on the hidden suite): 93% of oracle pass@16 kept on Llama-3.1-8B (0.848 vs 0.909, 7.7% visible false-positive rate, 3.04x cost cut) and 96% on Qwen2.5-Coder-7B (0.921 vs 0.963, 3.8% FP, 3.52x) (evals/RESULTS.md, "Robustness").

Why a skeptical buyer would believe it: the same seeds draw the same 16 samples in both arms, and early-stop only skips samples after a pass, so pass@16 is identical by construction (evals/RESULTS.md). Aborted partial tokens are counted. The naive baseline ran second on a warm prefix cache, a bias against the scheduler. The buyer can rerun the 194-line script. What they should not believe yet: any number on their repo, their model, or their test harness (section 6).

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| Solve API (`POST /v1/solve`: prompt, tests[], N, B, budget) | custom, extends dexa_platform/gateway/app.py (FastAPI, OpenAI-compatible) | request contract, auth, per-solve metering |
| Control plane (keys, credits, usage) | reuse dexa_platform/control (29 tests pass; 03-build-inventory.md) | tenancy, billing records |
| Search scheduler | custom; seed is the loop in evals/modal_verifier_sched.py | seed B samples per problem into one live batch; verify finished samples each step; abort siblings on pass; refill failed waves up to N |
| Verifier runners | custom orchestration over OSS sandboxes | run customer tests; today a subprocess with 10 s timeout |
| Engine | OSS: vLLM 0.24 LLMEngine (measured) or SGLang (measured under load) | prefill/decode; prefix caching shares the prompt KV across the B samples |
| Load harness | reuse voice-inference vkv/loadgen/runner.py, vkv/metrics/{events,manifest,analyze}.py, evals/modal_arm_a.py sweep driver | p95-gated concurrency sweeps, resumable Modal sweeps |

Where the differentiation lives: in the scheduler policy on the engine's step loop, and in the API contract (tests are a first-class input). Not in a kernel, not in the model, not in a KV connector. Everything below the scheduler is stock vLLM or SGLang; nothing is replaced. That is the honest reading of the ledger: the branched-decode "white space" was engine-owned (SGLang n=8 vs n=1 decode/step 2.2x at 128k vs vLLM 174x; 01-evidence-ledger-dexa.md, sglang-gate), and the large W x B win was RadixAttention prefix caching (1.00-1.02x vs a cached baseline; agentic-serving-value-v1).

```
 coding agent / harness
     |  POST /v1/solve {prompt, tests[], N, B, token_budget}
     v
 +-----------------------------+   keys, credits, per-solve metering
 | gateway (dexa_platform)     |   (reuse control/ + gateway/)
 +--------------+--------------+
                v
 +-----------------------------+      +-----------------------------+
 | search scheduler            |<---->| verifier pool               |
 |  seed B -> step -> verify   |      |  sandboxed test runners     |
 |  -> abort siblings on pass  |      |  (subprocess today)         |
 |  -> refill failed wave (<=N)|      +-----------------------------+
 +--------------+--------------+
                | add_request / step / abort_request
                v
 +-----------------------------+
 | engine: vLLM LLMEngine or   |  OSS, unmodified; prefix cache
 | SGLang                      |  shares the prompt KV
 +-----------------------------+
```

## 6. Evidence

### Proven

- Best-of-N with a unit-test verifier lifts HumanEval pass 0.689 -> 0.902 (N=1 -> 16) at 52k -> 858k tokens; flat on TriviaQA (0.680 -> 0.693) — 01-evidence-ledger-dexa.md, compute-allocation-map; evals/modal_eval_compute.py.
- Continuous scheduler: identical pass@16 (0.884), 2.63x fewer decode tokens, 2.12x faster end-to-end — 01-evidence-ledger-dexa.md, verifier-early-stop; evals/modal_verifier_sched.py.
- Weak verifier (first 3 asserts) keeps 93-96% of the oracle ceiling at 3.04-3.52x lower cost on two 7-8B models — 01-evidence-ledger-dexa.md, verifier-weak-robustness; evals/modal_robustness.py.

### Bounded / contradicted

- Under load on SGLang (W=4k, B=8, k=2), early-stop throughput is 1.38x at C=8, 1.69x at C=32, 1.87x at C=96 — not the 4x the token ratio predicts, because decode is bandwidth-bound; C=1's 4.0x is a cold-start artifact — 01-evidence-ledger-dexa.md, contention-v2-early-stop. Verification there is simulated by a pass probability, not real tests.
- On an idle GPU, early-stop saves 2-4x decode tokens but 1.00-1.02x GPU-seconds vs a prefix-cached baseline; the 11.5x headline is prefix caching — 01-evidence-ledger-dexa.md, agentic-serving-value-v1. Contradiction with the 2.12x wall-clock above: the sched run measures one shared flight on vLLM with a cold-cache bias against itself; agentic-value measures per-turn GPU-seconds on SGLang with simulated verification. Both are cited; neither has been run under the other's conditions.
- Round-based (client-replicable) early stop reached 2.60x tokens but 1.53x wall; continuous in-engine reached 2.63x and 2.12x — evals/RESULTS.md. These were separate runs with different naive baselines (168.1 s generation wall vs 183.8 s end-to-end), so the in-engine increment is bounded, not a controlled delta.
- Verdict recorded in the repo: "buyer-replicable, not a moat" and "a buyer captures most of the 1.85x with round-based early-stop in their own orchestration" — docs/FINDINGS.md; evals/RESULTS.md "Contention v2".
- pass@16 varies run to run at fixed seeds across scripts (0.909, 0.896/0.878, 0.884, 0.902) — evals/RESULTS.md; sampling noise, so quality comparisons are only valid within a run.
- vLLM 0.24 parallel sampling (n>1) over a long shared prefix is superlinear (n=8 decode/step 174x n=1 at 128k); SGLang is 2.2x — 01-evidence-ledger-dexa.md, vllm-decode-pathology and sglang-gate. The sched run used max_model_len 4096 and independent n=1 requests, so long-context branching for this product is unmeasured on either engine.
- Superseded, do not cite: the 2026-07-15 "2.8-4.7x cheaper at equal quality" product claim (evals/RESULTS.md "Product implication") is listed as superseded by agentic-value v1 and contention v2 in 04-conflicts.md. Token counts above are still measured; "cheaper" holds only for a token-billed buyer or under load.
- Unreconciled in the repo (04-conflicts.md, conflict 6): the 2026-08-02 text calls code-gen quality "the one large, durable lever" while FINDINGS.md marks early-stop "buyer-replicable, not a moat"; the repo then moved to a stateful-session thesis on 2026-08-05 without amending either.

### Unproven

| claim | experiment that proves or kills it | est. cost (our estimate; HumanEval sweep ran ~4.5 min of A100 per model-config per evals/RESULTS.md wall times) |
|---|---|---|
| Gain holds on repo-level tasks with real test harnesses (not HumanEval) | SWE-bench-Verified subset (100 tasks), N=16, B=4, sandboxed pytest as verifier; measure tokens, wall, pass@16 vs naive; record verify latency separately | ~40-80 A100-hours (test runs dominate), 10 calendar days |
| Gain holds at 30B-A3B class and above | rerun modal_verifier_sched.py on Qwen3.6-35B-A3B (OpenHands default) and one larger open MoE; same gates | ~10 GPU-hours, 3 days |
| Gain holds at 100K+ shared prefix on the chosen engine | 8-way branch over 32k/64k/128k prefixes as independent prefix-cached requests on vLLM and SGLang; decode/step n=8 vs n=1 | ~6 GPU-hours, 2 days (scripts: modal_decode_pathology.py, modal_sglang_pathology.py) |
| In-engine continuous abort beats the best client-side loop under load with real verification | one run, one engine, three arms (naive / client rounds over HTTP / in-engine), real tests, C in {8,32,96} | ~12 GPU-hours, 4 days |
| Verifier latency does not eat the wall-clock win | measure verify-on-critical-path share when tests take 5-60 s (repo tests) vs 10 ms (HumanEval) | included in row 1 |
| Customers' "visible tests" behave like first-3-asserts | replay 20 customer-provided test sets; measure FP rate and no-visible-pass rate | 0 GPU-hours for FP scoring after row 1 samples exist; 5 days of collection |
| Per-solve pricing clears GPU cost | instrument tokens and GPU-seconds per solve at C=32 on rented A100/H200 ($2.50/hr H200 per 05-research-kvcache.md) | derived from rows 1 and 4 |
| Any comparison to Fireworks/Together/Morph latency or cost | same prompts, same tests, client-side loop on their APIs vs ours | ~$200 of API spend (estimate), 2 days |

## 7. MVP and 6-week build plan

What ships first: a hosted `/v1/solve` endpoint on one model (Qwen2.5-Coder-7B-Instruct, the measured higher-ceiling model, 0.963 oracle pass@16) plus a public, rerunnable benchmark page reproducing section 4 and adding the SWE-bench-Verified subset result whatever it turns out to be.

- Week 1: lift the scheduler loop out of evals/modal_verifier_sched.py into a long-running Modal service; keep the LLMEngine step loop; add a request queue so many solves share one flight; reuse evals/modal_robustness.py `visible_check` as the customer-tests adapter. New: request schema, token budget cap, streaming of the passing sample.
- Week 2: verifier pool. Replace the in-process subprocess with per-request sandboxes; report verify latency in the response. New code.
- Week 3: gateway and billing. Mount the solve route in dexa_platform/gateway/app.py (stream=False is forced at app.py:127 per 03-build-inventory.md); wire dexa_platform/control for keys and credits; add per-solve metering (new; today metering is per token).
- Week 4: load harness. Port voice-inference vkv/loadgen + vkv/metrics to text prompts (the inventory lists what generalizing needs: tokenizer, chat endpoint, aiohttp abort, server-metrics scraping); run the C in {8,32,96} sweep with real verification.
- Week 5: the repo-level experiment (Unproven row 1) and the 30B-A3B rerun (row 2); section 10 gates decide what the launch page says.
- Week 6: launch page with rerunnable scripts, an OpenAI-compatible fallback (`n` plus `stop_on_pass` on /v1/chat/completions), and plugins in Morph's distribution shape (MCP, Vercel AI SDK, OpenRouter, Claude Code/OpenCode plugins; 05-research-morph.md).

Reused: evals/modal_verifier_sched.py (194 lines), modal_verifier_engine.py, modal_robustness.py, modal_contention.py, dexa_platform/gateway and control (29 tests), voice-inference vkv harness (27 tests with pytest-asyncio). New: sandboxed verifier pool, per-solve billing, streaming, the service wrapper. Not reused: src/dexa (KV persistence/compaction) or the Dexa vLLM connector.

## 8. Pricing model

The architecture makes the token bill go down (2.6x fewer decode tokens at equal quality), so per-token billing of the same API is self-cannibalizing for the provider and is precisely why generic per-token providers are not structurally motivated to ship it: a token-billed provider that adds verify-abort sells fewer tokens per solve. Three billable units the architecture can express:

1. Per solve, with a token budget: price = f(model, N cap, tokens actually decoded including aborted partials). Precedent for non-token units in the same market: Morph Router $0.005/request, Reflexes $0.001/event (05-research-morph.md).
2. Per token at a list rate near peers (Morph Fast Apply $0.80/$1.20; Relace $0.85/$1.25 per M; 05-research-morph.md) with early-stop as the reason the buyer's bill is lower than a client-side loop on the same rate.
3. Reserved GPU-hours with the scheduler included, for self-hosters who saw 1.38-1.87x more turns per GPU under load (01-evidence-ledger-dexa.md, contention-v2); Tensormesh's "30% of estimated savings" formula (05-research-kvcache.md) is an existing precedent for savings-share pricing.

What is unmeasured: our GPU-seconds per solve at production concurrency (Unproven rows 4 and 7). Until then no price point should be published; the founder picks the unit (section 11).

## 9. Competitive facts

| who | what they ship that is adjacent | what they do not ship (as far as the research files show) | source |
|---|---|---|---|
| Morph | Fast Apply 10,500 tok/s, WarpGrep, Compact, Reflexes (engine "forked from vLLM", custom kernels), Router, hosted Kimi K3/GLM-5.3/DeepSeek V4 with prefix caching, sticky KV, 50% standby tier; OpenAI/Anthropic-compatible; team size 3 | no test- or verifier-guided sampling endpoint in the catalog fetched | 05-research-morph.md (morphllm.com, docs, YC) |
| Relace | Apply 3 (LoRA SFT on 3-8B, FP8 via vLLM llm-compressor, spec decoding, 10k+ tok/s), FAS search subagent (RL, 8xH200 ~1.5 days) | no verifier-guided best-of-N endpoint | 05-research-morph.md (relace.ai blogs) |
| Cognition | SWE-grep/SWE-grep-mini on Cerebras (2,800 tok/s), SWE-1.6 at 950 tok/s in Windsurf | no third-party API or pricing for these in the files | 05-research-morph.md, 05-research-agents.md |
| Fireworks | prompt caching within one replica, 50% cached discount, session-affinity and multi-turn session headers for RL rollouts, adaptive speculative decoding (350 ms TTFT case) | no verify/abort scheduling or per-solve billing in the files | 05-research-agents.md, 05-research-voice.md |
| Wafer | Turbo models "2-2.8x faster than base SGLang/vLLM", Wafer Pass $10/$25 per week for Claude Code/OpenHands/Cline harnesses, dedicated endpoints, ZDR header | no verifier API | 05-research-wafer.md |
| Tensormesh | cached input billed $0, reserved H200 $2.50/hr, savings-share pricing | no verifier API | 05-research-kvcache.md |
| vLLM / SGLang (OSS) | n>1 sampling, prefix caching/RadixAttention, LLMEngine abort_request (the primitive the scheduler uses); SGLang handles n=8 at 128k at 2.2x/step | no built-in verify-abort-refill policy; the repo's scheduler drives it from outside | evals/RESULTS.md; 01-evidence-ledger-dexa.md |
| Anthropic / OpenAI | prompt caching (5 min/1 h; 30 min/24 h), Managed Agents at $0.08/session-hour, Codex/Claude Code harnesses | no verifier-guided sampling API in the pages fetched | 05-research-caching.md, 05-research-agents.md |

## 10. Risks and pre-registered kill gates

| risk | the measurement | kills | proceeds |
|---|---|---|---|
| HumanEval result does not transfer to repos | SWE-bench-Verified subset, tokens at equal pass@16 vs naive | < 1.5x | >= 2.0x (measured 2.63x on HumanEval) |
| Result does not transfer to bigger models | same on Qwen3.6-35B-A3B | < 1.5x | >= 2.0x |
| Real verify latency eats the wall-clock win | end-to-end speedup with sandboxed tests on the critical path | < 1.3x | >= 1.8x (measured 2.12x) |
| Customer visible tests are weaker than 3 asserts | % of oracle pass@16 kept on customer test sets | < 85% | >= 90% (measured 93-96%) |
| No throughput win under load | agent-turns/s ratio at C=32 with real verification | < 1.3x | >= 1.6x (measured 1.69x, simulated verifier) |
| In-engine adds nothing over a client-side loop | wall-clock, in-engine vs HTTP rounds, same engine, C=32 | < 1.15x: ship as an OSS scheduler/SDK, not a hosted engine | >= 1.3x (bounded: 1.53x -> 2.12x across two runs) |
| Long-prefix branching is pathological on the chosen engine | n=8 vs n=1 decode/step at 128k as prefix-cached independent requests | > 4x on both engines | <= 2.5x on at least one (SGLang measured 2.2x for n>1) |
| Unit economics | GPU-seconds per solve at C=32 on rented H200 vs list price of the chosen unit | gross margin < 0 at Morph-level token rates | unmeasured until rows 1 and 4 |

## 11. Founder decisions

- Hosted API vs OSS scheduler vs SDK vs all three. Evidence: the "in-engine vs client loop" gate; Morph and Relace are hosted and closed, LMCache/Tensormesh is OSS plus hosted (05-research-kvcache.md).
- Whether Morph/Relace/Cognition adjacency matters. Facts only: none ships a verifier endpoint in the files; all sell to the same buyer; Morph's catalog grows by adding narrow endpoints, and its published SWE-Bench Pro lifts are +2.1 to +3.7 points (05-research-morph.md).
- Model choice: 7-8B (measured), 30B-A3B (OpenHands default, unmeasured), or frontier open MoE (Kimi K2.7 Code, GLM-5.2; 05-research-agents.md). Evidence: Unproven row 2.
- GPU class: A100 (measured) vs H200/B200 vs MI355X (Wafer's numbers, 05-research-wafer.md). Nothing here is measured off A100.
- Engine: vLLM 0.24 (measured scheduler) vs SGLang (measured under load and at long prefixes) vs a fork. Evidence: sglang-gate and vllm-decode-pathology.
- Verifier scope: customer tests only, or also a learned verifier/classifier in the style of Morph Reflexes (05-research-morph.md). No evidence either way for learned verifiers in the repos.
- Pricing unit: per solve, per token, or savings-share (section 8).
- Market size and share of coding-agent spend that is generation vs search: not judged here; the files give search at 56.6% of tokens (Relace) and Anthropic's ~400k sessions with ~4 user turns and ~10 actions per prompt (05-research-agents.md).
- Whether to accept the ledger's "buyer-replicable" verdict as disqualifying, or as a statement about where to put the gate (section 10, row 6).

## 12. Combinations

- With the session-stateful / KV-tiering candidate (dexa stateful sessions; voice-inference arm D pinning): the coding-agent step has a 116-126K median prefix (05-research-agents.md) and search branches share it. Agentic-value v1 measured the prefix-cache win (up to 11.5x vs re-prefill) and the early-stop win as separate, additive levers (01-evidence-ledger-dexa.md); a provider keeping the prefix resident across idle gaps and searching within it combines both. Unmeasured beyond 32k and single-GPU.
- With a Morph/Relace-style specialized model: the scheduler is model-agnostic; a fine-tuned code model with a custom speculator (Relace's recipe, 05-research-morph.md) raises the ceiling (Qwen2.5-Coder reached 0.963 vs Llama's 0.909) and per-sample speed.
- With the benchmark-harness direction: the voice-inference load harness is the artifact section 7 week 4 needs.
- With computer-use / UI-testing (Morph Glance, dexa CUA gateway): the verifier becomes a UI test; nothing measured.