# Candidate E — Verifier-guided search inference for code ("pass@16 quality at pass@4 cost")

## 1. Thesis

A specialized inference endpoint for code: the request carries a prompt plus the customer's tests; an in-engine verify-abort-refill scheduler samples until one candidate passes. The customer sentence: "We send the prompt and the tests; we get back the first solution that passes and pay for the solve, not for sixteen tries."

Measured basis (one run, one seed, one 8B model, HumanEval): the continuous scheduler matched naive best-of-16 pass@16 exactly (0.884 = 0.884) at 2.63x fewer decode tokens (327,607 vs 862,050, aborted partials counted) and 2.12x less end-to-end wall (86.6 vs 183.8 s) (01-evidence-ledger-dexa.md, verifier-early-stop). The repo's own verdict, which this brief designs around rather than disputes: the token saving is "Real but buyer-replicable, not a moat" (docs/FINDINGS.md); "a buyer captures most of the 1.85x with round-based early-stop in their own orchestration" (evals/RESULTS.md "Contention v2"). So the product case rests on what a client loop cannot get on a token-billed API: (i) the prompt prefix prefilled and billed once per solve rather than per sample, (ii) mid-decode abort and refill inside one batch, (iii) per-solve pricing; plus unmeasured extensions (real repos, larger models, 100K prefixes) priced as experiments in section 6. Two caveats carried throughout: every headline number is single-seed (04-conflicts.md (f)); the 2.12x wall-clock is confounded by asymmetric synchronous verification (section 4).

## 2. Customer and workload

Who buys: teams shipping coding agents or code-generation features on open-weight models. Facts: OpenHands recommends Qwen3.6-35B-A3B on vLLM/SGLang, 32,768 context; Cursor's Composer 2 started from the open-weight Kimi K2.5 base, with Moonshot naming Fireworks (medium confidence, unconfirmed by Cursor); open-weight models carried a majority of OpenRouter tokens by mid-2026, "concentrated in coding and agentic workloads" (all 05-research-agents.md). Morph's founder-stated customers (create.xyz, databutton, continue.dev, Framer, Webflow, Block, Vercel; currency unverified; 05-research-morph.md) fit the profile.

What they run today: TraceLab (~4,300 Claude Code/Codex sessions, 43 developers): median cached prefix per step 126,180 / 115,584 tokens, ~857/886 new input, 252/184 output tokens, median human gap 1.4 min; Mooncake's Codex traces ~131:1 input:output (05-research-agents.md). Search is 56.6% of tokens in coding traces (Relace) and "60% of their turns" (Morph), vendor-stated (05-research-morph.md).

What the research files do not show: any coding-agent product running best-of-N with tests at serving time; the traces above are one sample per step. The documented "many samples plus a verifier" workload is RL rollouts: Fireworks' rollout-inference docs (sticky session routing, mid-stream weight swaps; 05-research-providers.md), Cursor's RL infrastructure with "hundreds of thousands of concurrent sandboxed coding environments" (05-research-agents.md), Relace training FAS with verl (05-research-morph.md). Serving-time demand is unmeasured; section 10 gates it, section 11 decides it.

How they pay today: per token on generic APIs (Fireworks: prefix caching within one replica, cached tokens ~50% off; 05-research-providers.md) or specialized ones (Morph Fast Apply $0.80/$1.20; Relace Apply 3 $0.85/$1.25 per M; 05-research-morph.md), or per GPU-hour (Modal A100-80GB ~$2.50/hr; 05-research-providers.md).

Measurement scope: Llama-3.1-8B-Instruct (in-engine) and Qwen2.5-Coder-7B-Instruct (post-hoc only) on A100-80GB, vLLM 0.24.0; load and long-prefix results on unpinned SGLang (01-evidence-ledger-dexa.md). Nothing at 30B-A3B or larger, off A100, or above max_model_len 4096.

## 3. The pain, in the customer's words

Quotes from the research files:
- "even claude is well over 11% error rates with search and replace" — Morph Launch HN (05-research-morph.md).
- "a specialized model like FAS is only useful if you can actually separate search from the rest of the agentic coding task" — Relace (05-research-morph.md). "Generate until the tests pass" is separable.
- ~$15.9 for ~20M tokens across 112 model calls in 4 turns when prompt caching failed on Azure OpenAI (05-research-agents.md, openai/codex #25604). Agent token bills are visible.

Our paraphrase, not a quote: "Best-of-16 with my tests takes this 8B model from 0.69 to 0.90 on HumanEval, but it costs ~16x tokens, so I don't run it in production." Measured lever: 0.689 (N=1, greedy) -> 0.829 (N=4) -> 0.902 (N=16), 52k -> 858k tokens; single model; oracle selector; "directional, not a leaderboard" (compute-allocation-map). The repo: "The cost is naive-16x, which is the opportunity" (evals/RESULTS.md).

## 4. Value proposition and the proof-of-value benchmark

Metric, at fixed N and B: pass@N on a held-out suite; decode tokens billed (aborted partials counted); GPU-step wall with verification off the critical path; end-to-end wall. The claim is equal pass@N at fewer tokens and less wall, never a higher pass rate.

Measured setup (evals/modal_verifier_sched.py): all 164 HumanEval problems, N=16, B=4, temperature 0.8, top_p 0.95, max_tokens 640, per-sample seeds 1..16 (seed 0 is the post-hoc scripts), unsloth/Llama-3.1-8B-Instruct, A100-80GB, vLLM 0.24.0 LLMEngine add_request/step/abort_request, max_model_len 4096, enforce_eager; verifier = check() in a `python -c` subprocess, 10 s timeout, synchronous inside the step loop.

Baselines, named: (a) same engine, 16 independent n=1 requests per problem, no early stop — measured; (b) round-based early stop, B=4 per llm.chat call, verify between rounds — measured (evals/modal_verifier_engine.py), the repo's client-loop proxy; (c) a streaming HTTP client loop that opens B requests and closes losers on first pass — unmeasured in either repo, and what a competent buyer would build; (d) SGLang prefix-cached serving (RadixAttention, agentic-value v1's "cached" column) — measured, simulated verifier; (e) Fireworks/Together/Morph client loops — unmeasured.

What was measured: continuous scheduler 0.884 vs 0.884; 327,607 vs 862,050 tokens (2.63x); 86.6 vs 183.8 s (2.12x). Round engine: 330,174 vs 858,658 tokens (2.60x); GPU generation wall 110.0 vs 168.1 s (1.53x, called "directional (prefix-cache state persists across calls)" in the script); verify wall 18.6 vs 27.0 s timed outside generation (evals/RESULTS.md "Live engine"). Weak-verifier numbers: section 6.

Why a skeptical buyer would believe the token number: both arms draw the same 16 per-sample seeds and early-stop only skips samples after a pass, so pass@16 is identical by construction; aborted partials are billed; the naive arm ran second on a warm prefix cache (a bias against the scheduler, stated only in the script docstring); the 194-line script is rerunnable.

Why the same buyer should not yet believe the 2.12x wall-clock (found in this revision by reading the script): the naive arm calls passes() on all 2,624 finished samples synchronously inside engine.step(); the scheduler arm verifies only finished samples of unsolved problems, never aborted siblings (modal_verifier_sched.py lines 120-139 vs 156-165). The engine does not step while a subprocess runs, so an unlogged, asymmetric amount of verification time sits inside both walls; per-verification latency and counts are recorded nowhere. The cleanest GPU-side number is the round engine's 1.53x generation wall, labeled "directional"; 2.12x is end-to-end with a confound. Section 10 pre-registers the decomposition. Not yet believable at all: any number on the buyer's repo, model, prefix length, or concurrency.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| Solve API (`POST /v1/solve`: prompt, tests[], N, B, budget) | custom; extends dexa_platform/gateway/app.py (FastAPI) | contract, auth, per-solve metering |
| Control plane (keys, credits, usage) | reuse dexa_platform/control (29 unit tests; no live-backend integration test; no payment path; 03-build-inventory.md) | tenancy, billing records |
| Search scheduler | custom; seed is the loop in evals/modal_verifier_sched.py | seed B samples per solve into one live batch; verify finished samples asynchronously; abort siblings on pass; refill failed waves to N |
| Verifier pool | custom orchestration over OSS sandboxes; today a synchronous subprocess, 10 s timeout | run customer tests off the GPU critical path; report verify latency |
| Engine | OSS, unmodified: vLLM 0.24 LLMEngine (measured) or SGLang (measured under load and at 128k) | prefill/decode; prefix cache shares the prompt KV across B samples |
| Load harness | reuse voice-inference vkv/loadgen/runner.py (57 lines), vkv/metrics/{events,manifest,analyze}.py (273 lines), evals/modal_arm_a.py sweep driver | p95-gated, resumable concurrency sweeps |

Where the differentiation lives: the scheduler policy on the engine's step loop and the API contract (tests as first-class input; prefix prefilled once per solve). Not in a kernel, model, or KV connector: branched decode is engine-owned (SGLang n=8 vs n=1 decode/step 2.2x at 128k vs vLLM 0.24's 174x; sglang-gate, vllm-decode-pathology) and the large W x B win was RadixAttention prefix caching (1.00-1.02x vs cached; agentic-serving-value-v1, FALSIFIED). Version dependency: the scheduler drives vLLM 0.24.0's synchronous LLMEngine loop; vLLM's latest release is 0.28.0 (05-research-kvcache.md); nothing was run on newer vLLM or pinned SGLang.

```
 agent / harness / RL rollout worker
     | POST /v1/solve {prompt, tests[], N, B, budget}
     v
 [gateway: dexa_platform control/ + gateway/, per-solve metering]
     v
 [search scheduler: seed B -> step -> verify -> abort siblings on pass
                    -> refill wave (<=N)]  <--async-->  [verifier pool:
                                                         sandboxed tests]
     | add_request / step / abort_request
     v
 [engine: vLLM LLMEngine or SGLang, unmodified; prefix cache shares prompt KV across B]
```

## 6. Evidence

### Proven
- Best-of-N with a unit-test verifier lifts HumanEval pass 0.689 -> 0.902 (N=1 greedy at temperature 0 -> N=16 at 0.8), 52k -> 858k tokens; flat on TriviaQA (0.680 -> 0.693); single model and seed; oracle selector — compute-allocation-map.
- Early-stop keeps the same pass@16 at ~2.6x fewer decode tokens, three ways in-engine on Llama-3.1-8B/HumanEval: post-hoc 2.85x (B=4), round engine 2.60x, continuous scheduler 2.63x at identical 0.884 — verifier-early-stop. Single-seed; ~1.7x Modal box-to-box wall variance noted (04-conflicts.md (f)).
- Weak verifier (first 3 asserts of check(), graded on the hidden suite, post-hoc over one n=16 seed-0 generation): Llama-3.1-8B 0.848 vs 0.909 oracle (93%), 7.7% visible false-positive rate, 3.04x cost cut; Qwen2.5-Coder-7B 0.921 vs 0.963 (96%), 3.8%, 3.52x; 6-10 problems per model with no visible pass within N; not a live run; the only Qwen2.5-Coder-7B measurement — verifier-weak-robustness.

### Bounded / contradicted
- Wall-clock: 1.53x GPU-generation wall (rounds, "directional") and 2.12x end-to-end (continuous, with section 4's verification asymmetry). The in-engine increment over rounds is bounded, not a controlled delta: separate runs, different baselines (168.1 s generation vs 183.8 s end-to-end).
- Idle SGLang GPU: early-stop saves 2-4x decode tokens but 1.00-1.02x GPU-seconds vs a prefix-cached baseline (W up to 32k, B up to 8); the 11.5x headline is prefix caching; simulated verifier; FALSIFIED — agentic-serving-value-v1.
- SGLang under load (W=4k, B=8, k=2, distinct per-agent contexts, simulated verifier): 1.38x at C=8, 1.69x at C=32, 1.87x at C=96, not the 4x the token ratio predicts (decode is bandwidth-bound); the C=1 point is a cold-start artifact the repo says to ignore — contention-v2-early-stop (BOUNDED). Contradiction with 2.12x: different engine, workload, verifier; neither run under the other's conditions.
- Repo verdicts (section 1) are unreconciled (04-conflicts.md, conflict 6): the same file calls code-gen quality "the one large, durable lever" and later adopts a stateful-session thesis without amending either.
- pass@16 differs across scripts (0.909, 0.896/0.878, 0.884, 0.902) because seeds and request shapes differ per script; quality is comparable only within a run.
- Long prefixes: vLLM 0.24 parallel sampling n=8 decode/step is 174x n=1 at 128k (eager, prefix caching off); SGLang 1.04x at 32k, 2.2x at 128k, unpinned. The scheduler ran at max_model_len 4096 with independent n=1 requests, so branching over 100K prefixes as prefix-cached independent requests is unmeasured on either engine. Llama-3.1-8B bf16 KV is ~131 KB/token and "8 x 128k contexts do not fit 80 GB HBM" (04-conflicts.md; context-scaling-branching): concurrent long-prefix solves per GPU are HBM-bound and unmeasured.
- Superseded, not cited: the 2026-07-15 "2.8-4.7x cheaper" claim (04-conflicts.md).

### Unproven
| claim | experiment | est. cost (ours; unmeasured) |
|---|---|---|
| Gain holds on repo-level tasks with real tests | SWE-bench-Verified subset (100 tasks), N=16, B=4, sandboxed pytest, 3 seeds; tokens, GPU-step wall, end-to-end, pass@16 vs naive; verify latency logged | tests dominate; ~2 weeks |
| Verification asymmetry explains part of the 2.12x | rerun modal_verifier_sched.py with verification in a thread pool; log verify count/time per arm; report GPU-step wall | ~1 A100-hour |
| In-engine beats a streaming client loop under load; GPU-seconds per solve | one engine, three arms (naive / HTTP streaming loop with close-on-pass / in-engine), real tests, C in {8,32,96} | ~12 GPU-hours, 4 days |
| Gain holds at 30B-A3B and above | rerun on Qwen3.6-35B-A3B and one larger open MoE | ~10 GPU-hours, 3 days |
| 100K+ prefix branching not pathological; concurrent solves per GPU | 8-way independent prefix-cached branches at 32k/64k/128k on current vLLM and SGLang; solves/GPU at a p95 SLA | ~6 GPU-hours, 2 days |
| Customer visible tests behave like first-3-asserts | replay customer test sets over existing samples; FP and no-visible-pass rates | collection only |
| Serving-time search is a workload customers run | customer discovery: teams running or willing to run best-of-N with tests in production | 0 GPU-hours |
| Any comparison to Fireworks/Together/Morph | same prompts and tests; client loop on their APIs vs ours | ~$200 API spend, 2 days |

## 7. MVP and 6-week build plan

What ships first: a hosted `/v1/solve` on one model plus a public, rerunnable benchmark page reproducing section 4 and reporting the SWE-bench subset and verification decomposition whatever they show. Model: Llama-3.1-8B-Instruct is the only model with an in-engine scheduler run; Qwen2.5-Coder-7B has the higher post-hoc ceiling (0.963) but none; the week-2 rerun decides.

- Week 1: lift the loop from evals/modal_verifier_sched.py into a long-running service; keep the LLMEngine step loop; move verification to a thread pool (new; today synchronous); add a request queue so many solves share one flight; log verify count/time; run the decomposition (Unproven row 2).
- Week 2: verifier pool: per-request sandboxes replacing the subprocess; verify latency in the response. New code: modal_robustness.py's visible_check parses HumanEval's check(), not arbitrary tests. Rerun the scheduler on Qwen2.5-Coder-7B.
- Week 3: gateway and billing: mount the route in dexa_platform/gateway/app.py (stream=False forced at app.py:127; add streaming); wire dexa_platform/control keys and credits; per-solve metering is new (today per-token; no payment path).
- Week 4: load harness: port voice-inference vkv/loadgen and vkv/metrics to text (gaps per 03-build-inventory.md: synthetic token-id prompts, no tokenizer, no mid-stream abort over urllib, no server-metric scraping); run the three-arm C in {8,32,96} sweep with real verification.
- Week 5: SWE-bench subset (row 1, 3 seeds) and the 30B-A3B rerun (row 4); section 10 gates decide the launch page.
- Week 6: launch page with rerunnable scripts; OpenAI-compatible fallback (`n` plus `stop_on_pass`; requires modifying the vLLM OpenAI server, no repo evidence); plugins in Morph's shape (MCP, Vercel AI SDK, OpenRouter, Claude Code/OpenCode; 05-research-morph.md).

Slip risks: SWE-bench sandboxing is the long pole (no repo precedent); async verification changes the measured loop; current vLLM is unmeasured. Reused: modal_verifier_sched.py (194 lines), modal_verifier_engine.py, modal_robustness.py, modal_contention.py, dexa_platform gateway/control (29 tests), vkv harness (27 tests, pytest-asyncio unlisted). Not reused: src/dexa KV persistence/compaction, the Dexa connector.

## 8. Pricing model

Structural facts. In-engine, a solve's B branches share one prefilled prefix; on an idle SGLang GPU an 8-way branch at 32k costs 1.02x the GPU-seconds of one cached turn (agentic-serving-value-v1). On a token-billed API each sample re-bills the cached prefix (Fireworks: cached tokens at ~50% per request; 05-research-providers.md), so a client loop pays the prefix per sample while an in-engine provider can bill it once per solve. That pricing capability is not replicable client-side; the token-count saving is (evals/RESULTS.md "Contention v2").

Three billable units:
1. Per solve with a token budget: f(model, N cap, tokens decoded including aborted partials, prefix billed once). Non-token precedents: Morph Router $0.005/request, Reflexes $0.001/event (05-research-morph.md).
2. Per token near peer rates (Morph $0.80/$1.20; Relace $0.85/$1.25 per M; 05-research-morph.md), early-stop being why the bill is lower than a client loop at the same rate.
3. Reserved GPU-hours with the scheduler, for self-hosters (1.38-1.87x turns/GPU under load; contention-v2); Tensormesh's post-v1 "30% of estimated savings" formula (FAQ; SaaS in beta; 05-research-kvcache.md) is a savings-share precedent.

Anchor (arithmetic, not a measurement): the scheduler arm's 86.6 s on one A100 at Modal's ~$2.50/hr (05-research-providers.md) is ~$0.06 for 164 HumanEval solves, ~$0.0004 per solve vs ~$0.0008 naive; single-tenant, short prompts, serialized verification included, boot excluded. GPU-seconds per solve at concurrency are unmeasured; no price point before that.

## 9. Competitive facts

| who | what they ship that is adjacent | what they do not ship per the research files | source |
|---|---|---|---|
| Morph | Fast Apply (self-reported 10,500 tok/s on B200; OpenRouter observed 1,537 tps), WarpGrep, Compact, Reflexes (engine "forked from vLLM", stated for Reflexes only), Router, hosted Kimi K3/GLM-5.3/DeepSeek V4 with prefix caching, sticky KV, 50% standby tier (GLM-5.3 only); YC team size 3, one person until ~2026-08-25 | no test- or verifier-guided sampling endpoint in the catalog fetched | 05-research-morph.md |
| Relace | Apply 3 (LoRA SFT on 3-8B, FP8 via vLLM llm-compressor, 10k+ tok/s), FAS search subagent (verl, 8xH200) | no verifier-guided best-of-N endpoint | 05-research-morph.md |
| Cognition | SWE-grep (>650 tok/s), SWE-grep-mini (>2,800 tok/s) on Cerebras; SWE-1.6 up to 950 tok/s in Windsurf | no third-party API or pricing in the files | 05-research-agents.md |
| Fireworks | prompt caching within one replica, ~50% cached discount, session-affinity and x-multi-turn-session-id headers for RL rollouts | no verify/abort scheduling or per-solve billing in the files | 05-research-providers.md |
| Wafer | Turbo models (company-stated): Qwen3.5-397B-Turbo 2.8x faster than base SGLang, GLM5.1-Turbo 2x faster than vLLM baseline; Wafer Pass $10/$25 per week (April 2026; current status unknown) for Claude Code/OpenHands/Cline; dedicated endpoints; ZDR header | no verifier API | 05-research-wafer.md |
| Tensormesh | cached input billed $0, reserved H200 $2.50/hr, savings-share formula post-v1 | no verifier API | 05-research-kvcache.md |
| vLLM / SGLang (OSS) | n>1 sampling, prefix caching/RadixAttention, LLMEngine abort_request; SGLang n=8 at 128k at 2.2x/step; vLLM latest 0.28.0 | no built-in verify-abort-refill policy; the repo drives it from outside | 01-evidence-ledger-dexa.md; 05-research-kvcache.md |
| Anthropic / OpenAI | prompt caching (5 min/1 h; 30 min/24 h), Managed Agents $0.08/session-hour, Codex/Claude Code harnesses | no verifier-guided sampling API in the pages fetched | 05-research-caching.md |

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| The 2.12x wall is partly a verification artifact | GPU-step wall ratio, verification off the step loop, 3 seeds | < 1.3x | >= 1.8x (rounds: 1.53x GPU-only) |
| In-engine adds nothing over a streaming client loop | wall and tokens, in-engine vs HTTP loop with close-on-pass, same engine, C=32, real tests | < 1.15x: ship as OSS scheduler/SDK | >= 1.3x |
| HumanEval does not transfer to repos or bigger models | SWE-bench subset, tokens at equal pass@16, 3 seeds, on Llama-3.1-8B and Qwen3.6-35B-A3B | < 1.5x | >= 2.0x (HumanEval 2.63x) |
| Real verify latency eats the win | end-to-end speedup, sandboxed tests, async verification | < 1.3x | >= 1.8x |
| Customer tests weaker than 3 asserts | % of oracle pass@16 kept on customer sets | < 85% | >= 90% (93-96% measured) |
| No throughput win under load with real tests | turns/s ratio at C=32 | < 1.3x | >= 1.6x (1.69x simulated) |
| Long-prefix branching pathological or HBM-bound | n=8 vs n=1 decode/step at 128k as prefix-cached requests; concurrent 100K-prefix solves/GPU at a p95 SLA | > 4x on both engines, or < 4 solves/GPU | <= 2.5x on one engine (SGLang 2.2x) |
| No serving-time demand | customer discovery count | founder-set | founder-set |
| Unit economics | GPU-seconds per solve at C=32 vs the chosen unit's list price | gross margin < 0 at Morph-level rates | unmeasured until the rows above |

## 11. Founder decisions

- Delivery form: hosted API vs OSS scheduler vs SDK vs all three. Evidence: the client-loop gate; Morph/Relace hosted and closed; LMCache/Tensormesh OSS plus hosted (05-research-kvcache.md).
- Which buyer: serving-time search (undocumented in the files) vs RL-rollout inference (documented: Fireworks rollout docs, Cursor RL infra, Relace verl) vs both; the discovery gate informs it.
- Whether to bill the prefix once per solve (section 8) and which unit: per solve, per token, GPU-hours, savings-share.
- Whether adjacency to Morph/Relace/Cognition matters. Facts: none ships a verifier endpoint; all sell to this buyer; Morph's self-evaluated WarpGrep SWE-Bench Pro lifts are +2.1 to +3.7 points (05-research-morph.md).
- Model: 7-8B (measured), 30B-A3B (OpenHands default, unmeasured), frontier open MoE (Kimi K2.7 Code, GLM-5.2; medium-confidence source). GPU class: A100 (measured) vs H100/H200/B200 vs MI355X ($2.29-$2.95/hr rental range Wafer cites; 05-research-wafer.md); more HBM raises concurrent long-prefix solves per GPU (unmeasured).
- Engine: current vLLM vs SGLang vs a fork placing verify/abort/refill in the scheduler.
- Verifier scope: customer tests only, a learned verifier (Morph Reflexes style; no repo evidence), or static pre-checks.
- Which public proof: HumanEval (measured, weak) now vs SWE-bench (unmeasured, credible) later; and whether the repo's "buyer-replicable" verdict is disqualifying or merely the location of the gate.
- Market size and the generation-vs-search share of coding-agent spend: not judged; the files give 56.6% search (Relace) and ~400k Claude Code sessions, ~4 user turns each (05-research-agents.md).

## 12. Combinations

- With the session-stateful / KV-tiering candidate: the coding step's 116-126K median prefix is what B branches share; prefix caching gave up to 11.5x GPU-seconds vs re-prefill while early-stop gave 2-4x decode tokens at 1.00-1.02x GPU-seconds on an idle GPU (agentic-serving-value-v1): additive for a token-billed buyer, not in GPU-seconds. Restore-vs-prefill is measured in-engine only to 16k (04-conflicts.md, conflict 11).
- With a Morph/Relace-style specialized model: the scheduler is model-agnostic; a fine-tuned model with a custom speculator (Relace recipe) raises the ceiling (Qwen2.5-Coder 0.963 vs Llama 0.909, post-hoc) and speed.
- With RL-rollout inference: the same loop is a rollout scheduler; Fireworks' rollout features are the adjacent facts.
- With the benchmark-harness direction: vkv loadgen/metrics are the week-4 artifact. With computer-use (Morph Glance, dexa CUA gateway): the verifier becomes a UI test; nothing measured.
