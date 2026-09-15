# A. Session-stateful inference provider for long-idle, long-context agents

*A hosted open-model provider whose billable unit is a durable warm agent session: KV kept resident across minute-to-hour gaps via client-hinted HBM -> RAM -> NVMe demotion over unforked vLLM, billed as residency plus delta tokens, proven by a public idle-gap-resume benchmark against the strongest OSS offload baseline.*

**Evidence grade C.** The restore-vs-re-prefill physics is measured in-engine but only single-request and only to 16k (vLLM 0.24 + LMCache, Qwen2.5-7B, A100: 10.0x/17.1x), is contradicted on a disk tier under 16-24-way concurrency (0.37x single, 1.6-2.2x worse p99), and the only RAM-tier concurrency data is voice: single seed, L40S, ~2.5k-token sessions, LMCache 0.3.2 on vLLM 0.9.2, BOUNDED in the ledger. The brief's actual differentiator - client-hinted, tier-spanning proactive demotion beating stock LRU/ARC offload at agent context and serving concurrency - has never been built or run in either repo (04-conflicts #3, #9) and would grade D on its own. Nothing is measured on H100/H200, at TP>1, at 32k+ in-engine on the model the numbers come from, or on real agent traces.

**Weeks to first credible public proof:** ~6-7 weeks to an A100-only IGR-1 (arms A / vLLM OffloadingConnector / LMCache 0.5.x / ours, 16k-64k, 3 seeds, burst arrivals, bitwise+logit correctness, raw logs published) with 2 engineers and ~$500-1,000 of Modal/RunPod GPU; the draft's full IGR-1 (H100 grid + H200 point, 128k, NVMe direct-IO, enforcing demotion) is ~9-12 weeks; the product surface the draft put in week 6 is weeks 7-9. A provider TTL probe (IGR-0) is publishable in week 1 for ~$100 of API spend but proves the pain, not our value. Assumptions: porting the voice connector from vLLM 0.9.2 to the pinned 0.28 KVConnectorBase_V1 takes <=2 weeks (no precedent in either repo), TP=1 only, Qwen2.5-7B or Llama-8B fp8-KV so the 32x64k resident set fits a >=200 GB host, no fork of vLLM.

---

## 1. Thesis

A hosted inference provider for open-weight models whose unit of service is a durable, warm agent session rather than a stateless request. The customer opens a session; the provider keeps its KV resident across idle gaps of minutes to hours (HBM while active, pinned host RAM, then NVMe, dropped past a measured break-even) and bills each turn as residency time plus the tokens actually added, instead of re-billing a ~120K-token prefix.

What is measured: inside a real vLLM stack, restoring evicted KV from CPU beats cold prefill 10.0x at 4k and 17.1x at 16k, single request (01 lmcache-offload-restore); a contiguous pinned-RAM connector held 300 concurrent ~2.5k-token sessions on an L40S where LMCache 0.3.2 collapsed, and a client idle hint that pins short-idle sessions moved that knee from 200 to 300 sessions at one seed (02 vi-contiguous-vs-chunked-transfer, vi-armD-predictive-pinning, both BOUNDED). What is not measured anywhere in either repo is the product's differentiator: no code proactively demotes an idle session's KV across tiers; the voice connector only delays freeing HBM and the dexa tiering policy is an advisory label (04-conflicts #3, #9; 03-build-inventory). No RAM-tier restore has been measured at 16k-64k under concurrency (04 #1, #4), and in-engine evidence for the model most numbers come from stops at 16k (04 #11). This brief exists to get that measured in six to ten weeks, against the strongest open-source offload baseline rather than against vanilla vLLM.

## 2. Customer and workload

Customer: teams running coding, computer-use, or human-in-the-loop agents on open-weight models, self-hosting on vLLM/SGLang (OpenHands recommends Qwen3.6-35B-A3B; Browser Use ships a 30B-A3B model; Cursor's Composer 2 started from Kimi K2.5) or buying from generalized providers. Open models carry a majority of OpenRouter tokens by mid-2026, concentrated in coding and agentic work (05-research-agents.md, Mozilla). No survey exists of what fraction of agent teams self-host versus buy (05-research-agents.md, Unknowns); that is a week-1 question for design partners, not an assumption.

Workload shape, all external (05-research-agents.md):

- Per step: median prefix 126,180 tokens (Claude Code) / 115,584 (Codex); median new input 857/886; median output 252/184 (TraceLab, arXiv 2606.30560). 219 production Claude Code sessions: median request 195K input / 317 output; 96% of requests reuse >=90% of input as verbatim prefix (llm-d GLM-5.2 post).
- Gaps: human thinking gap median 1.4 min, p90 20.6 min (TraceLab); tool-driven gaps median 5.2 s / P99 81.4 s (vLLM x Mooncake); inter-turn pauses median 2 s / p99 11.4 min (llm-d); the 99.9th-percentile Claude Code turn grew from under 25 min to over 45 min (Anthropic).
- Cache behaviour: global prefix hit 95.7% but 84.4% on user-initiated steps vs 97.5% on tool-result steps (TraceLab). That misses concentrate on human-gap turns is an inference from these two numbers, not a stated finding.
- Computer use adds ~1,000-1,800 input tokens per screenshot and ~4,500 for the toolset (Anthropic docs).

Two things the research does not settle and the brief must not assume: that open-model providers' caches are known to expire inside these gaps (retention is mostly undocumented; section 3), and that the hint an agent harness can send (tool call vs waiting on a human vs scheduled) predicts return time on real traffic. Both are week-1 measurements.

## 3. The pain, in the customer's words

- "After a gap longer than the TTL, the next request recomputes the full input and re-establishes the cache"; API-key users default to 5 minutes (https://code.claude.com/docs/en/prompt-caching).
- 17.1% overpayment ($949 on a Sonnet 4.6 sample, $1,581 on Opus 4.6) reconstructed from 119,866 Claude Code calls after writes moved from 1h to 5m in March 2026; closed "not planned" (https://github.com/anthropics/claude-code/issues/46829).
- Codex on Azure past ~150K tokens: cached_input_tokens = 0; one run ~$15.9 for ~20M tokens with 3.28M re-billed tokens = 71% of fresh input; no passthrough for prompt_cache_retention (https://github.com/openai/codex/issues/25604).
- "prompt_cache_key ... do not pin requests to a machine or guarantee a cache read hit" (OpenAI); "Cache entries can be evicted at any time due to server load or restarts" (xAI); Fireworks caching "only works within 1 replica" and lasts "at least several minutes... up to several hours" (05-research-caching.md, 05-research-providers.md).
- "At step 10 of a typical agent loop the model processes ~11,500 input tokens to act on ~200 new tokens" (Tensormesh blog).
- Orchestrators "know session structure and tool-gap duration that request-local LRU cannot observe" (SGLang RFC #27574).

Caveat the customer will raise: the loudest documented pain is on Claude and on Azure/OpenAI in-memory caching, which this provider cannot serve. For open-model providers the documented facts are OpenAI 24h retention at no charge, DeepSeek "a few hours to a few days", Groq 2h (GPT-OSS only), Cerebras 5 min guaranteed up to 1h, DeepInfra explicit 1h at 2x write (05-research-caching.md, 05-research-providers.md). TraceLab's p90 human gap of 20.6 min sits inside several of those windows. Whether open-model providers actually drop a 120K prefix across a 20-minute gap is unmeasured; IGR-0 measures it.

## 4. Value proposition and the proof-of-value benchmark

Proposition: your agent's session stays warm through gaps your provider's cache does not survive; resume is verified lossless; you pay residency plus delta tokens.

Two benchmarks, in order.

**IGR-0 (provider TTL probe), week 1, ~$100 of API spend.** For ~10 providers (Anthropic, OpenAI, DeepSeek, Fireworks, DeepInfra, Together, Groq, Cerebras, Tensormesh, Morph) send a 100K-token prefix, re-send it with an ~850-token delta after gaps of 1, 5, 15, 30, 60, 120 min, three trials each, and record cached_tokens and TTFT. Publishes, per provider and with raw responses, the pain the draft assumed. It does not prove our value; it establishes the market fact.

**IGR-1 (Idle-Gap Resume), weeks 3-6.** Harness: vLLM's own `benchmarks/multi_turn/benchmark_serving_multi_turn.py --max-active-conversations`, which docs/BENCHMARK_PLAN.md names as designed to force eviction and retrieval from the offload backend, with routing that defeats the local prefix cache or caching OFF (BENCHMARK_PLAN gotcha #1), plus one run with caching ON and live KV larger than the HBM pool. Sessions replay gaps sampled from TraceLab (median 1.4 min, p90 20.6 min) and llm-d (p99 11.4 min), with ~860 new-input / ~200-250 output tokens per turn and burst arrivals (8 sessions resuming inside 10 s), because aggregate tier bandwidth, not single-request latency, decided every concurrent result in the repos (04 #1).

Resident-set arithmetic the draft omitted: Llama-3.1-8B bf16 KV is 131 KB/token (docs/RESULTS.md persist bench), so 32 sessions x 64k = 269 GB and 128 x 128k = 2.1 TB; rental host RAM is 125-283 GB per GPU (RunPod) to 2,048 GB per 8-GPU node (CoreWeave) (05-research-gpu-pricing.md). Qwen2.5-7B is 57 KB/token (02 ledger), so 32 x 64k = 120 GB. Models: Qwen2.5-7B-Instruct bf16 (the model behind 10.0x/17.1x and all HF numbers) on a >=200 GB host, and Llama-3.1-8B with fp8 KV, whose native 128k context the repo already prefilled in-engine on vLLM 0.24 (32.68 s at 128k, eager: 01 context-scaling-branching), avoiding the YaRN crash that capped Qwen at 16k (04 #11). Grid: context {16k, 32k, 64k, 128k} x sessions {1, 8, 32}, 3 seeds, RAM budget stated per cell; cells whose resident set exceeds host RAM run on the NVMe tier and are labelled so. GPUs: A100-80GB (continuity), H100-80GB, one H200 point (04 #5).

Arms, versions pinned: **A** vanilla vLLM prefix caching; **B1** vLLM native OffloadingConnector, CPU tier sized to hold every session, LRU and ARC (docs.vllm.ai kv_offloading_usage); **B2** LMCache current release (0.5.4, Aug 2026), layerwise; **D** ours: contiguous pinned-RAM connector with async layer-pipelined load plus enforced idle-aware demotion RAM -> NVMe -> drop driven by client idle hints; **C** = B1 with filesystem secondary tier; **E** = D with fp8 KV.

Metrics: p50/p99 TTFT on resumes after gaps over 5 min, on the full-load window (voice whole-run p95 understated full-load p95 by 20-25 ms: 02 discrepancy #2); aggregate restore GB/s; GPU-seconds per turn; GB-hours held per tier; correctness as bitwise equality of every restored KV block against the saved bytes plus top-1 agreement over 64 greedy tokens and per-token KL against a cold prefill, with cold-vs-cold divergence measured first as the noise floor. Greedy-text identity is reported, not gated: the voice repo found cached-vs-full-prefill kernel paths cascade into divergent greedy text even when bytes are correct (02 lesson #9; evals/modal_kv_correctness.py).

Primary gate: **D vs best-of-B** at 64k, 32 sessions, equal RAM budget: p99 resume TTFT no worse than B and GB-hours held >= 30% lower at equal p99; or, where B degrades under burst, D p99 <= 0.5x B. Sanity gate, not a differentiator: D vs A >= 5x p99 at 64k (single-request in-engine is 17.1x at 16k; under 5x means the connector is broken). The draft gated on D vs A; at this grid point A's ~60 GB usable HBM holds ~7 Llama-8B or ~16 Qwen-7B 64k sessions, so most A resumes are cold prefills by construction and any CPU offload wins (vLLM's blog reports 2-22x for its own connector). That comparison measures the value of offload, which OSS ships, not the value of this product.

Why a skeptical buyer would believe it: independent harness; every version pinned (voice ran LMCache 0.3.2 on vLLM 0.9.2, dexa an unpinned LMCache on 0.24.0); raw events and manifests with connector provenance (every voice manifest records git_sha "unknown" and no connector field); two GPU generations; three seeds; a correctness column with archived output; the strongest OSS baseline configured per its own docs; dollars from the invoice, not the $1.80/hr constant in evals/stateful_cost_model.py.

## 5. Architecture

| name | custom or OSS | role |
|---|---|---|
| Session API (create / turn / resume / close, streaming) | custom; seed dexa_platform/sessions (383 LOC; non-streaming; max_tokens=16 hard-coded) | Session is the addressable primitive; carries idle hint and tier policy per turn |
| Session-to-replica affinity + admission | custom, new | Routes a session to the replica holding its KV; admits by resident-set budget (today a single fixed URL) |
| vLLM serving tier | OSS vLLM, unforked | Prefill/decode, paged KV, prefix cache for the active window |
| Session KV connector | custom; seed voice vkv/backends/voice_kv_connector.py (574 LOC) | Contiguous pinned-slab save/restore, blake2b per-block chain index, delayed-free pinning from idle hints |
| Tiering policy engine (enforcing) | custom, **new**; dexa_platform/sessions/tiering.py is a break-even calculator with no enforcement path | Demotes HBM -> RAM -> NVMe -> drop from measured break-evens and client hints |
| NVMe tier | OSS (vLLM OffloadingConnector filesystem tier or LMCache disk backend) | Residency for hour-scale gaps |
| Portable KV blob + correctness probe | custom; src/dexa/session (DEXAKV01) + voice evals/modal_kv_correctness.py (bitwise round-trip; no archived output) | Migration, failover, restore verification |
| Metering / keys / credits | custom; dexa_platform/control (620 LOC; no payments, no OAuth) | Residency GB-hours per tier + delta tokens |
| Benchmark harness | OSS vLLM multi-turn bench; optionally voice vkv/ (needs text prompts, chat endpoint, aiohttp abort, provenance, multi-seed: 03-build-inventory) | IGR-0/IGR-1 and the public leaderboard |

Where the differentiation lives: (1) the session-scoped API and residency metering; (2) a client-hinted, tier-spanning demotion policy; (3) the contiguous transfer path that held concurrency in the voice repo. Not in restore itself: LMCache, vLLM OffloadingConnector, Mooncake, Dynamo KVBM and SGLang HiCache all restore. Not in time-aware retention as such: Mooncake ships soft-pins (30 min default, 24 h max), Dynamo exposes cache_control-style TTL retention, Morph documents per-request TTL control, and SGLang RFC #27574 / Mooncake PR #2835 propose TTL-bounded pins (05-research-kvcache.md, 05-research-agents.md, 05-research-morph.md). Built on vLLM through the stock KVConnectorBase_V1 seam, whose docstring calls it "experimental and subject to change"; the seed connectors target 0.9.2 (voice) and 0.24.0 (dexa) while vLLM is at 0.28.0. Known seed limits: TP=1, fp16/bf16, non-MLA, prompt-KV only, synchronous loads, and a store evicts the entry it matched so sessions sharing a prefix evict each other (03-build-inventory), which agent sessions sharing a ~4.5k-token toolset would trigger every turn.

```
 client agent ──(session_id, delta tokens, idle_hint: tool|human|scheduled)──▶ Session API / affinity router
                                                                    │ metering: residency GB-hr + delta tok
                                                                    ▼
                    ┌──────── GPU replica (vLLM + session connector; TP=1 today) ────────┐
                    │  HBM: active sessions (prefix cache; delayed-free pin by hint)      │
                    │   │ demote (idle > t_hbm)  [NEW: nothing demotes today]             │
                    │   ▼                              ▲ restore: 22 GB/s single-request  │
                    │  pinned host RAM slab (content-addressed)   (LMCache log); ~1.5 GB/s │
                    └──────┬────────────────────────────▲──────── aggregate at N=300 voice┘
                           │ demote (idle > t_ram) [NEW] │ restore ~3 GB/s naive torch.load;
                           ▼                             │ direct-IO unmeasured
                        NVMe tier ──(idle > t_nvme)──▶ drop, re-prefill on return
```

## 6. Evidence

### Proven
- In-engine, single request, RAM tier: vLLM 0.24 + LMCache (version unpinned), Qwen2.5-7B, A100, prefix caching OFF: 4k 301 -> 30 ms (10.0x); 16k 1,320 -> 77 ms (17.1x); 16,384 tokens restored in 39.8 ms at 22.0 GB/s (01 lmcache-offload-restore). Only 4k and 16k.
- HF-level physics to 64k, Qwen2.5-7B: prefill 296 / 1,321 / 3,167 / 8,567 ms vs pinned-CPU restore 25 / 115 / 182 / 297 ms at 4k/16k/32k/64k (11.8x, 11.5x, 17.4x, 28.9x); NVMe via naive torch.load 246-1,228 ms (01 stateful-warm-session-hf). HF prefill is far slower than vLLM (Llama-8B/8k: 3.3-4.2 s HF vs 617 ms vLLM, docs/RESULTS.md, a different model), so engine-relative ratios are smaller; restore time is sub-linear in bytes, so part of the widening is fixed-overhead amortization (04 superseded list).
- Lossless, portable resume, two separate facts: KV saved by one vLLM process and loaded by another with identical output, OPT-125m and Qwen3-0.6B at 8k, TP=1 (01 cross-instance-resume); HFBackend persist bench identical_output=true at 8k-64k on Llama-3.1-8B (01 raw-kv-resume, identity column only).
- fp8/int8 KV at 0.5x bytes with 100% greedy agreement over 48 tokens, one passage, HF (01 kv-interchange-formats).
- 128k prefill runs in-engine on Llama-3.1-8B, vLLM 0.24, A100, eager: 32.68 s (01 context-scaling-branching). Not a restore result; it bounds the YaRN crash to the Qwen harness.
- End-to-end demo: 12k session 5.4 s cold -> 1.5 s warm (3.7x), one run, no raw artifact in the repo (01 stateful-session-service-live; 03-build-inventory).

### Bounded / contradicted
- Contiguous vs chunked under concurrency (02 vi-contiguous-vs-chunked-transfer, BOUNDED): VoiceKVConnector N=300 p95 316.8 ms whole-run / 336.8 ms full-load vs LMCache 0.3.2 default 25,997.7 ms / layerwise 6,025.1 ms; L40S, Qwen2.5-7B, ~2.5k-token sessions, seed 1, vLLM 0.9.2. Not a controlled comparison of transfer granularity; no per-transfer timings archived; the "~7 ms restore" is a docstring estimate; the "42 GB working set vs 26 GB pool" is doc arithmetic.
- Idle hints move a knee (02 vi-armD-predictive-pinning, BOUNDED): arm D N=300 p95 272.2 ms PASS vs arm A max passing N=200: 1.5x, single seed; 8.1 ms margin on the full-load window; 3 of 7 full-load buckets above 300 ms; the hint used the trace's own median constants; N=400 failed (34,048 ms). The mechanism is delayed HBM free, not demotion.
- Crossover is a property of tier bandwidth (04 #1): pinned RAM at ~9-22 GB/s crosses below 4k for 7-8B on A100 (derived); the dexa disk-backed connector at ~0.6-0.9 GB/s lost 0.37x at 8B/8k (1,681 vs 617 ms) and lost 1.6-2.2x at p99 under 16-24-way concurrency even with async load (01 connector-under-concurrency-sync-async). RAM tier at 16k-64k under load: never measured.
- "Three reproductions" (04 #3): two are in-GPU prefix caching and stock LMCache; the claimed delta, the tiering policy, is advisory (edge/README.md; sessions/tiering.py).
- Proactive demotion (04 #9, superseded list): the "copy + free at turn end, ~4 GB working set" design was described in voice RESULTS.md and never built or run; both LMCache and the custom connector leave HBM vLLM-managed.
- Cost model (04 #5): "2-6x cheaper" and break-evens (64k: HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr) are modeled at $1.80/hr, $0.006/GB-hr RAM, $0.0002/GB-hr NVMe from HF single-request timings (evals/stateful_cost_model.py). Warm break-even = prefill_s x usable_HBM_GB / (kv_GB x 3600): faster prefill shrinks it, larger HBM grows it. No H100/H200/B200 measurement exists; both repos name this the top threat.
- In-engine Qwen evidence stops at 16k (04 #11): >=32k crashed vLLM V1 with YaRN in three configs.
- 08-02 "every path inside the inference stack is closed for a small team" vs 08-05 "the thesis is a business" (04 #6): unreconciled; IGR-1 D vs B is the resolving test.

### Unproven
| claim | experiment | est. cost (estimates, unmeasured) |
|---|---|---|
| Open-model providers drop a 100K prefix across 5-60 min gaps | IGR-0: 10 providers x 6 gaps x 3 trials | ~1 week, ~$100 API |
| RAM-tier restore beats re-prefill at 16k-64k under 8-32 concurrency with burst arrivals | IGR-1 arms A/B1/B2/D, A100 | ~200 GPU-hr, ~$500 (Modal A100 $2.50/hr) |
| Client-hinted demotion holds fewer GB-hours than OffloadingConnector/LMCache at equal p99 | IGR-1 D vs B with real gap replay | included above |
| The hint (tool/human/scheduled) predicts return time on real agent traffic | partner harness logs; hint on/off ablation | staff time |
| Win survives H100/H200 (faster prefill, PCIe Gen5, more HBM) | IGR-1 H100 grid + H200 point | ~200 GPU-hr, ~$800 (Modal H100 $3.95/hr) + ~$50 |
| Restore in-engine at 128k with correctness | Llama-8B native 128k, chunked prefill on | ~2 GPU-days |
| NVMe direct-IO restore GB/s and break-even | O_DIRECT/GDS vs torch.load, 8-way concurrent | ~2 GPU-days |
| Connector works at TP>1 and on an MLA/MoE partner model | TP=2 port on one partner model; KV bytes/token measured | ~2 engineer-weeks + ~1 GPU-day |
| Buyers pay residency + delta; $ saving material at open-model prices | 3 design partners, 2 weeks, invoice $/session-hour | staff time + IGR-1 |

## 7. MVP and 6-week build plan

MVP: a session API over vLLM on rented A100/H100 with an enforcing RAM+NVMe tier, an idle hint on every turn, observed (not inferred) warm/cold reporting, residency metering, and published IGR-0/IGR-1 with raw logs. Realistic six-week scope with two engineers is the benchmark plus a thin API; the product surface the draft put in week 6 slips to weeks 7-9.

- **Week 1.** Run and publish IGR-0. Three design-partner conversations; instrument their harness to log per-turn gap and gap type. Pin vLLM 0.28, LMCache 0.5.4, torch; run benchmarks/vllm_connector_check.py against the pinned vLLM to learn the current KVConnectorBase_V1 surface. Decide harness: vLLM multi-turn bench first; generalize the voice harness only if it cannot replay gap distributions.
- **Weeks 2-3.** Connector v0: port voice_kv_connector.py (vLLM 0.9.2) to the pinned API; add async layer-pipelined load (dexa's ThreadPoolExecutor + per-load CUDA stream pattern in src/dexa/engine/vllm_connector.py is the reference), decode-KV save, tenant-scoped keys; fix matched-entry eviction so shared-prefix sessions coexist. Run evals/modal_kv_correctness.py and archive its output. Slip risk: the API port has no precedent in either repo.
- **Weeks 3-4.** IGR-1 on A100, arms A/B1/B2/D at 16k/32k/64k, 3 seeds, burst arrivals, per-cell RAM budget. First kill-gate read (section 10, rows 1-4).
- **Week 5.** Enforcing tiering: NVMe secondary tier; demotion driven by the hint protocol (arm D's ~40 connector lines + 6-line client hint is the seed: 03-build-inventory) and measured break-evens; GB-hour metering from connector counters. Arms C and E. 128k on Llama-8B.
- **Week 6.** H100 grid and one H200 point; direct-IO NVMe; publish IGR-1 with raw events, manifests and connector provenance.
- **Weeks 7-9 (beyond the draft).** Product surface: reuse dexa_platform/sessions/{service,store,tiering}.py and dexa_platform/control/*; add streaming (gateway forces stream=False; SessionDO and sessions/backend.py hard-code max_tokens 16), warm/cold observed from connector counters (today warm = not first_touch), affinity map, durable session store (today an in-process dict), tenant-isolated KV. Design partners on the endpoint.

Reused: voice connector and, if chosen, harness; serve/vllm_lmcache_backend.py (the config behind 10.0x/17.1x); evals/modal_lmcache_restore.py, modal_stateful_session.py, stateful_cost_model.py (rewired to invoice rates); src/dexa/session/{blob,store}.py; scripts/modal_bench_contention.py; benchmarks/vllm_connector_check.py. Not reused: src/dexa/engine/vllm_connector.py disk path (0.37x), compaction, cartridges, edge/ until a live backend exists. New: enforcing policy engine, NVMe tier, affinity router and admission, streaming session API, GB-hour metering, TP>1.

## 8. Pricing model

Residency + delta tokens, four meters: (1) delta input tokens at a list rate; (2) resident prefix tokens on resume at $0 (Tensormesh's shape) or a low read rate; (3) residency per tier: HBM free while active up to a short window; RAM and NVMe per GB-hour or per 1M-token-hour (Gemini bills explicit caches $0.50-$4.50 per 1M tokens per hour); sessions drop past a customer-set TTL; (4) output tokens at list.

Cost floor from cited rates: Llama-8B bf16 KV at 131 KB/token gives 8.4 GB at 64k and ~16.8 GB at 128k; host RAM at $0.0036-0.008/GiB-hr (AWS proxy / Modal) puts a 128k session at ~$0.06-0.13 per RAM-hour; NVMe at ~$0.0001/GB-hr at ~$0.002/hr; fp8 halves both (05-research-gpu-pricing.md). Qwen2.5-7B is 57 KB/token (3.76 GB at 64k).

The dollar case, derived and stated plainly: re-prefilling 128k on an A100 takes ~10-33 s of GPU time (32.68 s eager at 128k; ~13k tok/s non-eager at 8k) at $2.50/hr, roughly $0.007-0.023. Against $0.06-0.13 per RAM-hour, RAM residency saves money only for gaps shorter than roughly 3-20 minutes on 7-8B; NVMe residency pays for hours. At open-model list prices the avoided re-prefill is $0.0013 (64k on DeepInfra Llama-8B at $0.02/M) to $0.0064 (70B at $0.10/M). For small models the buyable unit is p99 resume latency and sessions-per-GPU, not dollars; the dollar case grows with model size (Kimi K3 at $3.00/M input on Fireworks: a 128k cold resume is ~$0.38) and is unmeasured. Comparators: Anthropic 1h write = 2x base (128k on Sonnet 5 at $2/M: $0.51 vs $0.26); DeepInfra 1h retention 2x write, 0.2x read; Managed Agents $0.08/session-hour while running. Willingness to pay per GB-hour is unmeasured.

## 9. Competitive facts

| who | what they ship that is adjacent | what they do not ship per the research files | source |
|---|---|---|---|
| Anthropic | 5-min cache refreshed on use; 1h TTL at 2x write; Managed Agents sessions at $0.08/session-hour while running | Retention beyond 1h; documentation of whether the model-side cache stays warm in an idle Managed Agents session | platform.claude.com pricing, prompt-caching, managed-agents |
| OpenAI | 30-min TTL (GPT-5.6+); 24h retention pre-5.6 with KV offloaded to GPU-local storage; previous_response_id | Machine pinning or guaranteed hits; a retention price | developers.openai.com prompt-caching, your-data |
| Google Gemini | Explicit caches billed per 1M-token-hour ($0.50-$4.50), 1h default TTL; 55-day Interactions state | Explicit caching inside the Interactions API | ai.google.dev caching, pricing, interactions |
| DeepSeek | Disk cache, free storage, cleared "within a few hours to a few days", hits ~3% of miss price | Guaranteed hits; documented eviction policy | api-docs.deepseek.com |
| Fireworks | Per-replica cache, x-session-affinity, 50% cached discount, "several minutes... up to several hours" | Cross-replica KV movement; a retention SLA | docs.fireworks.ai prompt-caching |
| DeepInfra | Explicit 5m/1h retention at 1.25x/2x write, 0.2x read, 8,192-token increments, two models | Retention beyond 1h; tiered pricing | docs.deepinfra.com prompt-cache-retention |
| Groq / Cerebras | Groq: automatic, 50% discount, 2h expiry, GPT-OSS only; Cerebras: 5-min guaranteed up to 1h, no discount | Session pinning; retention beyond 2h | console.groq.com, inference-docs.cerebras.ai |
| Baseten | KV-aware routing over a real-time index of worker cache contents; cached input at 10% | Documented TTL, session affinity, retention pricing | docs.baseten.co bis-llm |
| Morph | Automatic prefix caching "with per-request TTL control", sticky placement via x-session-id, 50% standby tier; open models at 1M context | Published TTL bounds; tier pricing | docs.morphllm.com fast-models |
| Wafer | Wafer Pass flat $10-25/week; cached-input rates; dedicated endpoints; agent-harness integrations | A documented session/retention primitive | docs.wafer.ai, 05-research-wafer.md |
| Tensormesh | LMCache-based SaaS, cached input $0, Operator "coming soon", post-v1 price = 30% of estimated savings; $24.5M raised | Idle-window or TTL eviction (LMCache documents LRU/IsolatedLRU/noop only); named customers | tensormesh.ai pricing, faq; docs.lmcache.ai |
| LMCache / Mooncake / Dynamo KVBM / vLLM OffloadingConnector / llm-d / SGLang HiCache | Multi-tier offload (CPU, NVMe, S3, P2P); LRU/ARC/LFU/S3FIFO eviction; Mooncake soft-pin 30 min / 24h max; Dynamo cache_control-style TTL retention; OffloadingConnector 2-22x TTFT; llm-d 48K pull 235 ms vs 1,988 ms | A hosted multi-tenant session product; client-hinted idle-aware demotion (SGLang RFC #27574, Mooncake PR #2835 are open/draft) | github.com LMCache, Mooncake, ai-dynamo, llm-d, vllm docs, sglang issues 27574/24656 |
| Continuum; "Stateful Inference" (research) | TTL-pinned KV from reload cost and queueing delay, >8x JCT; persistent KV across turns, 2.1-4.2x per turn | A product or hosted service | arXiv 2511.02230, 2605.26289 |

## 10. Risks and pre-registered kill gates

| risk | measurement | number that kills | number that proceeds |
|---|---|---|---|
| Policy and transfer path add nothing over best OSS offload | IGR-1 D vs B1/B2, 64k x 32, equal RAM, p99 and GB-hr, 3 seeds | D within 10% of the better B on both | D p99 <= B and GB-hr >= 30% lower, or D p99 <= 0.5x B under burst |
| RAM-tier restore does not beat re-prefill at agent context under load | D vs A p99, 64k x 32, A100 | < 2x | >= 5x |
| Resident set exceeds host RAM so the real tier is NVMe | per-cell KV bytes; NVMe direct-IO GB/s, 8-way concurrent | NVMe resume slower than re-prefill at 64k on H100 | >= 3 GB/s and NVMe resume p99 <= 0.5x cold |
| Restore not verifiably lossless | bitwise KV equality; top-1/KL vs cold, with cold-vs-cold floor | any bitwise mismatch | 100% bitwise; agreement >= cold-vs-cold floor |
| Bigger/faster GPU moves the wall (04 #5) | H100 D-vs-A ratio relative to A100; H200 point | H100 ratio < 2x | >= 60% of A100 ratio and >= 3x absolute |
| KVConnector API port fails or drifts | vllm_connector_check.py on pinned 0.28; correctness archived | not passing by end of week 3 | 10/10 hooks + correctness archived |
| In-engine 128k does not run | Llama-8B, chunked prefill on | not running by week 5 | 128k resume measured with correctness column |
| Customers' gaps are inside providers' windows | IGR-0 + partner gap histograms | providers keep 100K prefixes >= 30 min AND all 3 partners p90 gap < 5 min | one provider drops at <= 15 min OR one partner p50 gap > 5 min |
| Hint does not predict return | hint on/off ablation; partner logs | hint-on hit rate within 5 pts of hint-off | >= 15 pts higher |
| TP>1 / MLA needed and not built | partner model KV layout; TP=2 restore | partner requires TP >= 2 and no path by week 9 | TP=2 restore measured with correctness |
| Nobody buys residency | 3 design partners, 2 weeks | 0 of 3 accept metered residency | >= 2 of 3 run production traffic |

## 11. Founder decisions

1. **Model class.** All repo evidence is 7-8B dense at TP=1. Customers' agents run 30B-A3B to 2.8T MoE (Kimi K3, GLM-5.2/5.3; llm-d notes MLA KV is 576 FP8 elements/token and TP=8 keeps eight KV copies). Options: prove on 7-8B then scale; start on a partner's model. Evidence: KV bytes/token and prefill tok/s for the target model; IGR-1 on it.
2. **Hosted provider vs operator/plugin.** README (Apache-2.0 sidecar, hosted SaaS out of scope) contradicts FINDINGS/platform/edge (hosted PLG) (04 #7). Tensormesh's Operator is "coming soon"; SGLang and Mooncake are adding TTL-bounded pins in the open. Options: hosted only; operator for self-hosters; both. Evidence: whether design partners self-host (no public survey exists).
3. **Substrate.** LMCache (10-17x single-request; collapsed at N=300 on 0.3.2), custom connector (held N=300), vLLM native OffloadingConnector (LRU/ARC; FS/S3/P2P tiers; TieringManager RFC open), SGLang HiCache (L2 private per instance). Evidence: IGR-1 arms B1/B2 vs D on pinned current versions.
4. **Frontier-model customers.** The loudest pain (section 3) is on Claude/Codex, which this provider cannot serve. Options: teams already on open models; teams converting because of TTL economics. Evidence: IGR-0; partner conversations; OpenRouter/Mozilla share data.
5. **Pricing shape.** Per-GB-hour, per-1M-token-hour (Gemini), $0 cached with per-token markup (Tensormesh), flat subscription (Wafer Pass $10-25/week). Evidence: partner reaction; measured $/session-hour from IGR-1.
6. **Compute sourcing.** Per-second rental (Modal A100 $2.50, H100 $3.95, H200 $4.54/hr; RunPod H100 $2.69-3.29) vs commit (SF Compute $1.90 average). Host RAM per GPU (125-283 GB rental; 2,048 GB per 8-GPU CoreWeave node) bounds resident sessions.
7. **Stay in the stack or go up-stack.** The repo's 08-02 verdict says up-stack; its 08-05 verdict says this is a business (04 #6). Evidence: IGR-1's D-vs-B result is the deciding number.
8. **Which differentiator to prove first.** (a) transfer-path robustness under burst concurrency, (b) client-hinted demotion measured in GB-hours/dollars, (c) the session API and metering on top of stock OSS offload. Each has a different first experiment and a different competitor set.
9. **Latency or dollars as the buyable unit.** On 7-8B the avoided re-prefill is ~$0.001-0.02 and RAM residency ~$0.06-0.13/hr; the dollar case needs large-KV models. This decides the proof model and the pricing meter.
10. **Upstream or proprietary hint protocol.** Upstream the hint envelope to SGLang RFC #27574 / vLLM TieringManager RFC #38260 / Mooncake PR #2835 and sell the hosted policy, or keep the protocol private. Evidence: RFC status (open/draft); partner preference.
11. **Correctness contract.** Bitwise KV equality (cheap, verifiable) vs statistical output agreement vs greedy-text identity (which the voice repo found is the wrong test). This is copy the buyer reads.
12. **First public artifact.** IGR-0 (week 1, ~$100, establishes the pain) before IGR-1, or IGR-1 only.
13. **Harness and thresholds.** vLLM's multi-turn bench (independent) vs the voice harness (owned, six changes); and the pre-registered numbers (5x, 30%, 0.5x, 15 points) are the drafter's, to be set by the founder before any run.
14. **Single replica vs affinity router in the MVP.** The router has no seed code and decides whether a session survives replica failure or scale events.

## 12. Combinations

- **With the voice candidate.** The arm D hint protocol is the same mechanism at 2.5k tokens and ~25 s idle; here it is 100k+ tokens and 1-20 min idle. One connector, one policy engine, two hint calibrations; the voice repo's N=300 result is this brief's only concurrency evidence.
- **With the portable KV blob (README direction).** Cross-instance identical resume (proven at TP=1 on small models) becomes migration and failover between replicas and a customer-exportable session artifact.
- **With fp8 KV (kv-interchange-formats).** Halves residency bytes and, per the cost model's formula, roughly doubles every break-even; arm E measures it against a baseline that also gets fp8.
- **With a Morph-style specialized model.** A compaction or fast-apply subagent served inside the same warm session avoids re-shipping the prefix to a second provider; Morph's Compact (33,000 tok/s, line deletion, byte-identical survivors) shows the shape.
- **With SGLang RFC #27574 / Mooncake PR #2835.** Upstream the hint envelope (retention: prefix_tokens, ttl_seconds) so the client protocol is standard and the hosted policy plus session metering is the product.
