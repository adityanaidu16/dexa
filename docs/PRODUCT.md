# What to build: the product decision document

*Compiled 2026-09-03 from two experiment repositories (`adityanaidu16/dexa`, `adityanaidu16/voice-inference`), their raw run logs, and same-day web research. Every number below traces to a file under `docs/product/context/` or to a URL recorded there.*

---

## 0. How to read this

**The rule this document follows.** It does not decide that a market is too small, and it does not decide that a direction is dead because someone else ships something adjacent. Those judgments are yours. Where such a judgment would normally appear, the document states the facts on both sides and files the judgment under *Founder decisions* (section 7) with the evidence that would inform it.

**What it does decide.** Whether a number is measured, modeled, bounded, contradicted, or merely designed; what an honest proof-of-value benchmark for each direction would look like; what exists in code today; and what the first six weeks of building would prove.

**Status vocabulary** (used for every measured claim):

| label | meaning |
|---|---|
| PROVEN | measured, with setup recorded, usually reproduced more than one way |
| BOUNDED | measured, but a ceiling or a strong caveat limits the claim |
| CONTRADICTED | two measurements in the repos point different ways; both are cited |
| FALSIFIED | the thesis as stated was tested and failed |
| OPEN | designed or planned, not measured |

**Where the detail lives.**

| path | contents |
|---|---|
| `docs/product/context/01-evidence-ledger-dexa.md` | 38 entries, one per tested thesis in the dexa repo |
| `docs/product/context/02-evidence-ledger-voice.md` | 21 entries for the voice repo, plus 13 doc-vs-data discrepancies |
| `docs/product/context/03-build-inventory.md` | every component in both repos, its real state, tests run |
| `docs/product/context/04-conflicts.md` | 14 contradictions with reconciliations, and a superseded-claims list |
| `docs/product/context/05-research-*.md` | external facts with URLs: Morph, Wafer, providers, caching, KV ecosystem, voice, agents, document AI, GPU pricing |
| `docs/product/briefs/*.md` | the eleven candidate briefs, each fact-checked against the ledgers, attacked by a skeptic pass, and revised; `_meta.json` carries each brief's evidence grade and weeks-to-proof |

---

## 1. The short version

**What the experiments established.** Across roughly thirty tested theses, most inference-efficiency ideas measured on real GPUs collapsed to about 2x, to a configuration change any customer can copy, or to a wall. Two mechanisms held up as *capabilities* rather than speedups, and one held up as a capacity win:

- **Restoring a session's KV cache from pinned CPU memory beats re-computing it**, by 10x at 4k tokens and 17x at 16k inside a real vLLM stack, single request (PROVEN); no in-engine restore has been measured above 16k. The same claim measured through a disk-backed store under concurrency **lost** to vLLM's batched re-prefill (0.37x at 8k). The reconciliation is arithmetic: restore wins when the tier's bandwidth times the engine's prefill time exceeds the KV bytes. Pinned RAM at 10 to 20 GB/s wins from about 4k tokens; a disk path under 1 GB/s wins only near 64k. **Nobody has measured the RAM tier at 16k to 64k under concurrency**, which is exactly the regime a long-context agent product would live in.
- **A session's KV state can be saved by one vLLM process and resumed by another with prefill skipped** (PROVEN at tensor-parallel size 1, but thin: identical output is recorded on a 125M-parameter model over two blocks, and 8,176 of 8,192 tokens on a 0.6B model at 8k; the lossless 8k-to-64k result on an 8B model is HuggingFace-level, not inside the serving engine). That is a durability and portability property, not a latency property.
- **Telling the serving engine when a session will return moved a concurrency cliff.** On one L40S with a 7B model, baseline vLLM served 200 concurrent voice sessions inside a 300 ms p95 first-token budget and then collapsed. Pinning sessions predicted to return within 25 seconds, and letting the rest evict into a fast restore path, held 300 (1.5x). Single seed, synthetic traces, 46 GB GPU, and below the 2x line the experiment pre-registered (BOUNDED).

**What the outside world looks like today** (facts, sources in section 4): every shipping KV-cache layer evicts with an LRU-family policy that does not know when a session will return; the SGLang proposals that would carry such hints are open or inactive; no first-party provider sells guaranteed session-pinned KV; frontier caches expire in 5 minutes to 24 hours; voice platforms price by concurrent line and one of them states it over-provisions GPU headroom to avoid queueing; production coding-agent traces show median prefixes of 115k to 126k tokens and human gaps with a median of 1.4 minutes and a p90 of 20.6 minutes. Morph reached a reported seven-figure run rate as a one-person company by training a narrow model and building its own engine for it; Wafer went from a kernel-optimization tool to an AMD-based inference provider with a $40M Series A in fourteen months.

**The eleven candidates.** Section 5 fleshes out each: (A) a session-stateful provider for long-idle agents; (B) conversation-native voice inference sold on sessions-per-GPU; (C) one session-aware residency scheduler for every idle regime, sold hosted, BYOC, and as engagements; (D) a document-VLM provider; (E) verifier-guided search inference for code; (F) a portable, governed KV-state layer as open-core infrastructure; (G) cartridges, a context compiler; (H) a computer-use agent endpoint; (I) the Morph playbook applied to document parsing; (J) preemptible-GPU inference made safe by portable state; (K) a benchmark-first "sessions-per-GPU" standard plus a resident-slot SLA.

**What they have in common.** Seven of the eleven (A, B, C, F, J, K, and the session half of H) share one mechanism: a KV connector that owns residency decisions the engine's LRU cannot make. They differ in workload, customer, and business shape, not in the core code. The other four (D, E, G, I) are model- or scheduler-side bets with different evidence profiles.

**The decisions only you can make** are listed in section 7. The ones that change the build order most: voice-first or agent-first; hosted endpoint, BYOC software, or the Wafer-style engagement motion; whether adjacency from Tensormesh, llm-d, NVIDIA Dynamo, and the SGLang RFCs is a threat or a set of baselines; and which GPU class the proof runs on, because both repos name "a bigger-HBM GPU moves the wall" as their top unmeasured threat.

**The experiments that inform the most paths at once** (section 8): the restore-versus-re-prefill crossover surface (context by concurrency by tier) on one model and one GPU; the same voice sweep on an 80 GB GPU; and a three-seed replication of the 1.5x. Roughly 60 to 100 GPU-hours in total.

---

## 2. What the experiments actually established

This section digests the two evidence ledgers. Setups are stated because they bound what can be claimed. Full entries with sources are in `docs/product/context/01-*.md` and `02-*.md`.

### 2.1 Efficiency theses that collapsed or hit a wall (dexa repo)

| thesis | result | status |
|---|---|---|
| Verifier-guided early stop for code (abort losing samples mid-decode) | Identical pass@16 at about 2.6x fewer decode tokens on HumanEval with an 8B model, reproduced three ways in-engine (2.85x post-hoc, 2.60x round engine, 2.63x continuous scheduler). The 2.12x end-to-end wall-clock is confounded: the naive arm verified all 2,624 finished samples synchronously inside the engine step loop while the scheduler arm verified far fewer, so the cleaner GPU-only number is the round engine's 1.53x, which its own docstring calls directional. Against a prefix-cached baseline on an idle SGLang GPU the saving is 1.00 to 1.02x GPU-seconds; under load it is 1.38 to 1.87x with a simulated verifier. Most of the token saving is capturable client-side with round-based sampling on any provider. | PROVEN token saving; wall-clock BOUNDED; GPU-seconds on an idle GPU FALSIFIED; the repo's own verdict is "buyer-replicable, not a moat" |
| Long-context branched decode is pathological | vLLM 0.24 parallel sampling at n=8 costs 174x a single decode step at 128k. SGLang costs 2.2x. | FALSIFIED as an industry gap; it is a vLLM-specific inefficiency |
| KV cache portability across numeric formats | fp8 and int8 halve the bytes with 100% greedy agreement over 48 tokens; int4 diverges at token 14 | PROVEN (one passage) |
| KV cache portability across fine-tunes | Base-to-instruct injection diverges within 2 to 5 tokens despite 0.98 cosine similarity; a learned linear bridge recovered 6% | FALSIFIED |
| Agentic serving beats generic serving | Up to 11.5x vs. no-cache re-prefill, but 1.00 to 1.02x vs. a prefix-cached baseline; the win is prefix caching, which every serious engine ships | FALSIFIED as differentiation |
| Open VLM matches GPT-4o on documents at lower cost | Qwen2.5-VL-7B 0.925 vs. GPT-4o 0.880 on the same 200 DocVQA pages, relaxed match, one run; the GPT-4o run's output is unrecorded except that number, GPT-4o-mini is unrecorded, and GPT-4o is a 2024 comparator. Cutting the image budget from 1536px to 1024px gave 2.7x throughput at a 1-point loss, measured as one offline batch of 200 images at 32 output tokens (not a served endpoint, not full-page parse), and the ledger calls the resize one "a client can apply before any API". The widely quoted "40x cheaper" is a model: 3.4x fewer image tokens times an *assumed* $0.20 per million token price, and its 56-pixels-per-token formula contradicts the repo's own measured prompt sizes (1,017 and 2,201 tokens including question and template), which collapse it to about 9 to 10x at the same assumed rates. On 200 pages the 1-point loss is 2 pages and the GPT-4o gap is 9. | Accuracy BOUNDED on 200 pages; throughput BOUNDED; cost MODELED |
| Content-aware visual-token pruning as a moat | Blocked by Qwen2.5-VL's positional encoding, which expects the full image grid; no accuracy numbers | OPEN |
| Computer-use screenshot reuse | Screens are 86% unchanged patch-to-patch (7.3x headroom) on one synthetic 15-step browser trajectory, but the vision encoder spills small changes across 3.4x more tokens; the realized ceiling is 2.3x on 8 frames, maybe 3x. No grounding accuracy has been measured at the 312 to 322-token screenshot budget, and on the document sweep the nearest budgets lost 20 points (277 tokens) and 49 points (182 tokens) | BOUNDED at ~2.3x; accuracy at the served budget OPEN |
| KV compaction (Attention Matching vs. H2O) | At the compression ratios people use (8x to 128x), H2O is better and 5x cheaper; Attention Matching wins only past 512x on multi-fact recall, with a mediocre ceiling | BOUNDED |
| Cartridges (train a compact KV for a static corpus) | The one attempt, on a 360M model on CPU, did not beat no-context; the GPU experiment with corpus-conditioned synthetic Q&A was never run; the compiler's position layout (query positions starting at the logical corpus length) cannot be expressed through vLLM's connector API without a model-runner patch or retraining at contiguous positions | OPEN, one negative data point |
| Mutable, versioned KV state (incremental recompute on mid-context edits) | 4.6x fewer tokens reprocessed on a simulated loop with a tiny random model; correctness unit-tested; no GPU wall-time measurement | BOUNDED |

### 2.2 The stateful-session thesis (dexa repo)

| measurement | number | setup |
|---|---|---|
| Restore vs. re-prefill, raw tensor copies | 11.8x at 4k, 11.5x at 16k, 17.4x at 32k, 28.9x at 64k | Qwen2.5-7B, A100-80GB, HuggingFace tensors, single request. Note: the re-prefill side is HF eager prefill, several times slower than vLLM's |
| vLLM prefix-cache hit vs. cold prefill | ~12x at 4k, 25 to 34x at 16k | vLLM 0.24, in-GPU hit, no transfer. This is the mechanism vLLM and SGLang already ship |
| LMCache CPU offload then restore, prefix cache off | 10.0x at 4k (301 to 30 ms), 17.1x at 16k (1,320 to 77 ms); 0.875 GB moved at 22 GB/s | vLLM 0.24 + LMCache, A100, single request. The harness crashed at 32k and above (Qwen2.5-7B with YaRN), so in-engine *restore* evidence stops at 16k; the engine itself ran Llama-3.1-8B at 128k in the branching experiment |
| Residency cost model | 2 to 6x cheaper at 2 to 120 minute idle gaps for a 64k session; break-even idle at 64k of ~2 min in HBM, ~11 min in RAM, ~4.9 hr on NVMe | Modeled from the HF numbers at assumed rates ($1.80/GPU-hr, $0.006/GB-hr RAM, $0.0002/GB-hr NVMe). Not measured. The deployed tiering policy is advisory only |
| Dexa's own connector, raw KV from a Modal volume | 0.37x vs. vLLM re-prefill at 8B/8k single request; p99 TTFT 1.6 to 2.2x worse than vanilla under 16 to 24 concurrent sessions even with async loading; adaptive policy matches vanilla by declining to load | Llama-3.1-8B, A100, ~0.6 to 0.9 GB/s effective load bandwidth |
| Cross-instance resume | Two separate vLLM processes; the second loads the first's KV and skips prefill. Identical output is recorded on OPT-125m (two blocks); 8,176 of 8,192 tokens matched on Qwen3-0.6B at 8k (next-token correct); the bit-identical 8k to 64k result on Llama-3.1-8B is HF-level | TP=1, single attention backend, single-step prefill only. Chunked prefill is not saved; decode-token KV is never saved; TP>1 is not implemented |
| Independent benchmark (`vllm bench serve`, prefix sharing) | The connector was 5.5x slower than a no-cache baseline because it keys on the whole prompt and gets zero hits on shared prefixes | OPT-125m, A10G |

**The contradiction and its resolution** (conflict 1 in `04-conflicts.md`). The 10 to 34x and the 0.37x are both real. Written next to each number, the restore tier's effective bandwidth explains everything: pinned RAM at 10 to 20 GB/s crosses over below 4k tokens; a Modal-volume disk path at under 1 GB/s crosses over near 64k. What no run in either repo measured: the RAM tier at 16k to 64k under concurrent load, on the same model and GPU. That single experiment is the pivot for candidates A, C, F, J, and K.

**Two more things the ledger insists on.** First, of the "three independent reproductions" the findings memo cites, one is vLLM's own prefix cache and one is stock LMCache; the claimed delta over LMCache, the economic tiering policy, is designed, modeled, and not enforced anywhere in the code. Second, restore evidence above 16k inside a production engine does not exist; the 32k and 64k restore points are HF-level, and the README's "100k+ token coding agent" workload has never been measured as a restore in-engine (the engine did prefill Llama-3.1-8B at 128k in the branching runs, at 32.7 s per prefill).

### 2.3 The voice experiment (voice-inference repo)

Rig: Modal L40S (46 GB), vLLM 0.9.2, Qwen2.5-7B fp16, synthetic conversation traces with 10% barge-ins, seed 1, real time. Metric: maximum concurrent sessions at p95 first-token latency at or below 300 ms.

| arm | mechanism | max passing N | p95 at N=300 |
|---|---|---|---|
| A: vLLM with prefix caching | HBM only | 200 | collapse (prefix-hit rate fell from 92% to under 2%) |
| B1: LMCache 0.3.2 | reactive CPU tiering, chunked loads | 200 | 26.0 s (6.0 s with layerwise loading) |
| B2: custom connector | same strategy, contiguous session-granular transfers through a pinned slab | 200 | 316.8 ms, stable, 17 ms over the gate |
| C: audio-triggered prefetch | warmup request fired at speech onset | 200 | 15.0 s (a collapse; the server also died at minute 41 of that run, so 37% of turns have no recorded latency) |
| D: predictive pinning | each request carries its predicted idle window; short-idle sessions pinned in HBM | **300** | **272.2 ms** |

What the ledger adds to the memo you already have:

- The 1.5x is one seed, one GPU, one passing point, and it is 300 divided by 200: no N between 200 and 300 or between 300 and 400 was run, so the true ratio is bracketed somewhere in [1.0x, 2.0x). Over the full-load window (minutes 5 to 40) the p95 is 291.9 ms, not 272.2 ms, and 3 of 7 five-minute buckets exceeded 300 ms. N=400 failed within ten minutes.
- The only offload baseline measured is LMCache 0.3.2 on vLLM 0.9.2. vLLM's native OffloadingConnector (in-tree since 0.11, LRU or ARC), LMCache 0.5.x with layerwise loading, and SGLang HiCache were never run, so the comparison does not yet isolate the pinning policy from a stale baseline.
- The idle-window hint is computed from the synthetic trace's own median constants, and the 25-second pin threshold pins essentially every reply of 76 tokens or fewer (the median reply is 75). Whether the gain comes from prediction or from "pin every short reply" needs a shuffled-hint ablation, which was not run.
- Restores run synchronously inside the scheduler step and already cost about 24 ms of p95 at N=200 with 140 MB sessions on a 7B model; per-restore timings were never captured. A code-reading finding (a store evicts the slab entry it just matched, so a new session's first turn can evict a sibling sharing its system prompt) may have depressed the recorded 35 to 52% hit rates.
- The mechanism the original plan specified, freeing idle sessions' HBM at turn end to bound the working set, was never built. What was built and measured is cheap eviction plus selective pinning. The stock KV-connector API can delay a block's free (pin) and manage the connector's own tier, but it cannot order HBM eviction or prefetch without a request; proactive turn-end offload, demotion, and "enforced tiering" therefore need either vLLM's OffloadingConnector policy hook, the proposed TieringManager, or a fork.
- The whole-run p95 for the connector without hints was 316.8 ms; on the full-load window it was 336.8 ms. The arm-D advantage at N=300 is 44.6 ms of p95 over the same connector without hints, which makes the 1.5x an SLO-threshold artifact in one specific sense: at any SLO between 350 and 400 ms both configurations hold N=300 and the ratio is 1.0x. A published SLO curve, rather than a single 300 ms number, is what would show where the policy matters.
- Pinning holds about 70 sessions inside the 10 GB HBM budget; every retained session beyond that rides the RAM-tier restore path, which is the configuration that failed the 300 ms gate at N=300 without hints. A retention guarantee at scale therefore rests on the arm that failed, not the arm that passed.
- The audio stream contributed nothing measurable in any arm; the pinning signal is response length plus turn-taking statistics.
- The "7 ms restore" is a docstring estimate; no per-restore timings were archived. Server-side hit rates quoted in the write-up are not archived either.
- The 95% idle figure comes from synthetic traces computed with a different model's KV geometry and a decode speed higher than the one measured on the GPU.
- Both repos name the same top threat and neither measured it: on an 80 GB GPU the baseline's wall likely moves past voice-realistic session counts. The arithmetic in `04-conflicts.md` puts the baseline cliff at roughly 400 to 450 sessions on an 80 GB part for these traces.

### 2.4 Claims that must not be cited

The superseded-claims list at the end of `04-conflicts.md` has 27 entries. The ones most likely to leak into a pitch: "14 to 25x faster resume" (CPU, tiny model); "3.7 to 5.5x faster than re-prefill" (versus HF eager, not vLLM); "latency win is unconditional" (single request only); "40x cheaper than GPT-4o" (modeled); "7.3x fewer visual tokens" (superseded by the 2.3x ceiling); "the thesis is a business" (the differentiating policy is unmeasured); "PROCEED_3X" (the measured outcome was 1.5x with the tested mechanism at 1.0x).

---

## 3. What exists in code

From `03-build-inventory.md`; test suites were re-run on 2026-09-03.

**dexa** (22,460 lines of Python, 449 of TypeScript, 177 of Rust; 136 tests pass and 18 skip for missing torch/vLLM; 29 platform tests pass).

| component | state | what it is good for |
|---|---|---|
| `src/dexa/engine/vllm_connector.py` (917 lines) | runnable; validated on vLLM 0.24 at TP=1 | The lifecycle skeleton for a vLLM V1 KV connector with async loading and an adaptive load-or-recompute policy. Gaps: whole-prompt keying, no chunked-prefill save, no TP>1 |
| `src/dexa/session/` (533 lines) | runnable, tested | A bf16-native, mmap-loadable portable KV format, bit-identical at 8k to 64k on an 8B model when measured through the HF backend, not vLLM. The header carries shapes, dtype, and token ids; no weights hash, TP layout, or tenant field |
| `dexa_platform/gateway` + `control` + `sessions` (~1,700 lines) | runnable demo, 29 tests | OpenAI-compatible gateway with per-request cost telemetry, hashed API keys, credit ledger, a session API whose tiering decision is advisory. Streaming is forced off; billing, OAuth, and Redis are absent |
| `edge/` (449 lines TS) | typechecks; never deployed | Cloudflare Workers plus Durable Objects front door, one DO per session |
| `serve/*.py` | runnable on Modal | Three `vllm serve` launch configs: a screenshot-tuned Qwen2.5-VL, a document VLM, and vLLM+LMCache with a 40 GB CPU tier (no disk tier) |
| `evals/*.py` (24 scripts) | one-shot Modal experiments | The reproducible measurement scripts behind every number in section 2 |
| `src/dexa/compaction`, `cartridge`, `segment`, `memory` (~3,200 lines) | runnable at HF level; not wired to any serving path | Compaction benchmark harness; incremental-recompute planner; cartridge compiler with unproven quality |
| `native/kvcodec` | does not compile (one borrow-checker error) | nothing yet |

**voice-inference** (3,183 lines of Python in the package; 27 tests pass once `pytest-asyncio` is installed, which is missing from the requirements; ~265 MB of committed raw run logs covering 287,015 measured turns).

| component | state | what it is good for |
|---|---|---|
| `vkv/backends/voice_kv_connector.py` (574 lines) | ran on GPU per the archived runs; pinned to vLLM 0.9.2; no unit tests | The reference implementation of session-granular pinned-RAM tiering with contiguous transfers, async saves with backpressure, and hint-driven pinning through the stock connector API. The pinning logic is about 40 lines plus a 6-line client hint. Loads are synchronous; TP=1; fp16 only; prompt KV only |
| `vkv/{traces,orchestrator,loadgen,metrics,run,sweep}` (~2,000 lines) | working | A multi-turn conversational load harness with turn-taking, barge-in, evidence-grade flags, per-point archiving, and knee search. Generalizing it needs text prompts, a chat endpoint, mid-stream abort, and server-metric scraping |
| `evals/modal_arm_a.py` | working | A resumable serverless-GPU sweep driver |

**What a production inference provider would still need, whichever direction is chosen:** streaming, billing, auth beyond a hashed key, tensor-parallel support in the connector, chunked-prefill persistence, an enforced (not advisory) tiering policy, a session-to-replica router, tenant-namespaced KV keys, observability, and rate limiting. None of it exists. The two connectors are pinned to vLLM 0.9.2 and 0.24 against a current 0.28 whose connector hooks, constructor arity, and request-data shape have all changed; the verifiers put the port at two to three weeks, not one. Neither repo contains SGLang connector code, and SGLang's storage interface has no pin or delay-free hook today.

---

## 4. The outside world, as facts

Digest of the nine research files. Confidence and URLs are in `05-research-*.md`.

### 4.1 The two playbooks you named

**Morph** (`05-research-morph.md`). Legal entity AutoInfra, YC S23; the current product started in February 2025 after the founder studied why Cursor's fast-apply felt fast. Fast Apply merges code edits at a claimed 10,500 tokens per second and 98% accuracy on B200s using custom CUDA kernels and a task-shaped speculative decoder ("roughly 70 or 80% of the content is almost exactly the same, so you're essentially using the original code as a guess"; "we're almost making our own inference engine just for this task"). The catalog grew by adding narrow endpoints: WarpGrep (an RL-trained code-search subagent), Compact (33,000 tokens per second line-deletion compaction), Reflexes (multi-head trace classifiers on a vLLM fork with decode removed), a Router, and hosted open models with sticky KV placement and a half-price standby tier. Self-reported traction: $1.1M in five months, then a $6M to $7M run rate as a solo founder, and "0 to 10M of revenue" as a one-person company per a job post; team size 3; no funding beyond YC in primary sources. Peers Relace and Cognition publish the same recipe (LoRA fine-tune of a 3 to 8B base, FP8, speculative decoding, RL-trained search subagents on fast hardware).

**Wafer** (`05-research-wafer.md`). YC S25, founded 2025 as Herdora ("Cursor for CUDA"). It pivoted from a kernel-optimization agent and profiling tools to a hosted inference provider on AMD MI355X in about a year: OpenAI- and Anthropic-compatible serverless, dedicated endpoints with continual post-deployment optimization, and optimization engagements for clouds (DigitalOcean, Parasail). $4M seed in April 2026, $40M Series A on 2026-09-01 (Marathon, Chemistry, AMD Ventures; The Information reports a $200M+ valuation and acquisition offers). Eight people on the team page. An unlabeled logo carousel on its homepage shows Vapi, Tavus, Inworld, Vercel, and Ollama; whether they are customers, partners, or channels is not stated. Its stated thesis: GPU utilization averages about 20%, a few hundred people can hand-optimize accelerators, and "Nvidia's dominance is about software."

### 4.2 Generalized providers and their caches

Fireworks, Together, and Baseten publish proprietary or heavily modified engines (FireAttention kernels; Together's Turbo engine with speculators; Baseten's TensorRT-LLM builder and a vLLM-backed MoE stack with KV-aware routing). DeepInfra, Novita, Hyperbolic, Nebius, Parasail, RunPod, and Modal do not name a production engine. Fireworks closed a $1.5B Series D at $17.5B in July 2026 and states that over 95% of its tokens come from models specialized on customer data. Caching: Fireworks is automatic with a 50% discount, retention "several minutes up to several hours", and session-affinity headers because the cache lives in one replica; Groq expires after 2 hours; Cerebras guarantees 5 minutes with no discount; DeepInfra sells explicit 5-minute or 1-hour retention at 1.25x or 2x the input price to write. No provider surveyed sells guaranteed session-pinned KV.

### 4.3 Frontier labs

Anthropic: 5-minute default TTL, 1-hour option at 2x write cost, reads at 0.1x; a community audit attributed a 17% cost increase in Claude Code to a switch from 1-hour to 5-minute writes. OpenAI: 30-minute minimum on GPT-5.6+, 24-hour retention on earlier models (KV offloaded to GPU-local storage), and `prompt_cache_key` that "influences routing but does not pin." Gemini: explicit caches billed per token-hour of storage ($0.50 to $4.50 per million tokens per hour). DeepSeek: disk cache cleared "within a few hours to a few days." Anthropic's Managed Agents bill $0.08 per session-hour while running and nothing while idle, the first time-priced session primitive in the survey.

### 4.4 The KV-cache infrastructure ecosystem

LMCache (Apache-2.0, 11.6k stars) and its company Tensormesh ($24.5M raised; SaaS in beta billing cached input at $0; an on-prem Operator "coming soon"; post-v1 pricing 30% of estimated savings). Mooncake Store (approximate LRU, 10-second leases, soft pins up to 24 hours; a draft PR for TTL-bounded retention leases). NVIDIA Dynamo's KVBM (four tiers, presence/LFU offload filters; a BlueField-4 flash tier announced for 2H 2026). llm-d (CNCF sandbox; precise KV-event routing, tiered offload via vLLM's native OffloadingConnector with LRU or ARC, P2P KV pull, session affinity; powers GKE Inference Gateway). SGLang HiCache (private per-instance host memory; two RFCs for agent-aware and programmatic KV hints with pin, demote, prefetch, and TTL, both open or inactive). Production trace studies: Alibaba's ideal hit ratios 54 to 62% with a p99 KV lifespan of 97 seconds; llm-d's 219 Claude Code sessions with 96% of requests reusing at least 90% of their input verbatim and inter-turn pauses from 2 seconds median to 11.4 minutes p99. The Continuum paper pins GPU KV with a TTL derived from reload cost and reports over 8x job-completion improvements on agent benchmarks. **No shipping system in the survey evicts on predicted return time.**

### 4.5 Voice AI infrastructure

Three sourcing patterns: pass-through to frontier APIs (Vapi, Retell, Pipecat), self-hosted open models (LiveKit serves Gemma 4 31B on SGLang at 192 ms TTFT and states it reserves "more headroom than a throughput-maximized deployment would"; Inworld, ElevenLabs, Deepgram host 26B to 397B open models), and speech-native models. Concurrency is priced per line (Vapi $10 per line per month beyond 10; Retell $8 beyond 20). Volumes: Vapi 1 to 5 million calls per day; ElevenLabs 10 million conversations per week. The LLM's share of a voice minute ranges from under 5% (Inworld's self-hosted model) to about 65% (Retell's frontier lines). Latency budgets put LLM first-token at 375 ms target and 750 ms maximum. Nobody publishes sessions-per-GPU at a first-token SLO for the LLM stage, although adjacent public agent-trace benchmarks exist (vLLM's Mooncake trace replay, llm-d's Claude Code traces, VAST and Backend.AI's long-context runs); hosted endpoints run through the same client would yield a knee at requests-per-minute with an unknown GPU count behind it, a different quantity.

### 4.6 Long-running agents

TraceLab (4,300 Claude Code and Codex sessions): median prefix 115k to 126k tokens per step, ~860 new tokens, human gaps median 1.4 minutes and p90 20.6 minutes, tool gaps 5.2 seconds median and 81 seconds p99, 95.7% prefix-cache hits overall but 84.4% on human-initiated steps. Across every published gap distribution (TraceLab, llm-d's 2-second median and 11.4-minute p99, SJTU/Alibaba's 80% of reuses within 10 minutes and 97-second p99 KV lifespan), hour-scale idle is the p90 to p99 tail, not the body. Anthropic: the 99.9th-percentile Claude Code turn grew from under 25 to over 45 minutes. A Codex user on Azure saw total cache misses past 150k tokens and 71% of input re-billed. Durable-execution runtimes (Temporal, Inngest, LangGraph) wait indefinitely at no compute cost; the model-side cache is what expires. Open-weight models carried a majority of OpenRouter tokens by mid-2026, concentrated in coding and agents; Cursor's Composer 2 started from Kimi K2.5; OpenHands recommends Qwen3.6-35B-A3B on vLLM or SGLang with a 22k-token minimum and a 32k context window, which is the only open-weight agent context guidance in the research files; the 115k to 195k prefixes come from Claude Code and Codex traces on frontier models.

### 4.7 Document AI

Cloud OCR is $1.50 per thousand pages; extraction $30 to $50 per thousand. Reducto launched r-1 at $0.01 per page on 2026-09-01. Gemini bills a PDF page at 560 tokens (about $0.42 per thousand pages of input). DeepSeek's V4-flash-vision (2026-08-21) caps an image at 384 tokens at V4-Flash prices, about $0.17 per thousand images at peak and $0.08 off-peak (derived, accuracy unmeasured); gpt-5.6-luna lands near $0.25 per thousand pages. Google Document AI has no question-answering endpoint (Textract Queries and Azure query fields do), so a DocVQA score for it needs an OCR-plus-LLM pipeline. Transcription-style outputs run 1,500 to 3,000 tokens per page per Claude's PDF documentation, against the 32-token answers every repo measurement used. Qwen3-VL-8B reports DocVQA 96.1 versus GPT-5 at 91.5 in Qwen's own tables. On olmOCR-Bench the published bars are Nanonets OCR-3 at 87.4 (leaderboard, medium confidence), Mistral OCR 4 at 85.2, dots.mocr at 83.9, and Chandra at 83.1. The olmOCR synthetic-data recipe uses a frontier teacher at about $0.12 per page. N-gram and prompt-lookup draft models ship in Baseten's BIS-LLM, Together's ATLAS, and vLLM itself; DeepSeek-OCR and InternVL3.5 compress vision tokens at the model level. vLLM ships video-token pruning only; the image-pruning RFC is open with no maintainer comment. Together lists Qwen3-VL-32B at a DocVQA score 3.6 points below the model report, which is the kind of discrepancy a same-pages benchmark would settle.

### 4.8 GPU pricing (inputs to any residency model)

H100 on-demand from $1.73 (Vast) to $12.29 (Azure list) per GPU-hour, median $3.39 across neoclouds; H100 Spot $2.15 to $2.59 on Nebius and AWS; B200 $6.25 to $8.60 on-demand; L40S $0.99 to $2.25; A100-80GB $1.39 to $2.79. Host RAM proxies $0.0036 to $0.008 per GiB-hour; local NVMe about $0.0001 per GB-hour. Spot notice: AWS 2 minutes best-effort, GCP 0 seconds by default, Azure 30 seconds. The cost model in the dexa repo assumed $1.80 per GPU-hour and $0.006 per GB-hour of RAM.

---
