# Candidate G — Cartridges: the context compiler

## 1. Thesis

Compile a large static corpus (a repo, a doc set, a knowledge base) offline into a small trained KV artifact — a *cartridge* — that loads into stock vLLM as a precomputed prefix. Queries then attend over the corpus without re-prefilling it, without holding the full-corpus KV resident, and without the corpus having to fit in the model's context window. The artifact is a file: versioned, shareable, movable across replicas, and independent of any provider's prompt-cache TTL.

The honest state of this direction, in one line: everything about the *systems* half is built and unit-tested; everything about the *quality* half is unproven, and the one attempt so far (360M model, CPU) was negative. `04-conflicts.md` lists the product framing in `docs/CARTRIDGES.md` ("matches full-context quality at RAG cost, ~100x less memory") as **superseded/unsupported**. Nothing in this brief cites that framing as fact. The plan below is a two-week GPU experiment with a pre-registered kill gate, then a four-week build on whatever survives.

## 2. Customer and workload

**Who.** Teams whose agents or applications ask many questions of the same large, slowly changing text: a coding-agent vendor with per-repo context; a product with a per-tenant knowledge base; a compliance/legal team over a regulatory corpus. Cartridges are per-corpus artifacts, which fits the measured shape of reuse in production caches: on Alibaba's Tongyi traces, 10% of KV blocks contribute 77% of reuses and cross-user reuse is "very low" (05-research-kvcache, arXiv 2506.02634).

**Workload facts (external, not cartridge-specific).** Agent steps carry a median cached prefix of 126,180 tokens (Claude Code) / 115,584 (Codex) and append ~860 new tokens per step (TraceLab, arXiv 2606.30560). In 219 production Claude Code sessions, 96% of requests reused ≥90% of input as a verbatim prefix (llm-d GLM-5.2 post). Morph's WarpGrep docs claim coding agents spend "60% of their turns searching for code" (05-research-morph). Server-side compaction at Anthropic triggers at a default 150,000 input tokens (05-research-agents).

**The regime cartridges target.** High reuse (hundreds to thousands of queries per corpus version), static content (a tagged release, a doc snapshot), and corpora that are either too large to keep resident per replica or larger than the model window.

**Unmeasured.** No customer discovery exists for this candidate; corpus-size and query-rate distributions for any target segment are unknown. Proposed experiment: ten discovery calls with coding-agent and KB-product teams, collecting corpus size (tokens), refresh cadence, and queries per corpus-day. Cost: founder time, no GPU.

## 3. The pain, in the customer's words

There are no cartridge customers, so these are public statements from adjacent users of prompt caching — the incumbent solution a cartridge would replace. All quotes are verbatim from the research files.

- OpenAI Codex issue #25604 (Azure): long sessions past ~150K tokens "suffer total prompt-cache misses (cached_input_tokens = 0)"; one run re-billed 3.28M tokens, "71% of fresh input" (05-research-agents).
- Claude Code issue #46829: a TTL change from 1h to 5m produced a computed "17.1% overpayment ($949 on a Sonnet 4.6 sample)" across 119,866 calls (05-research-caching).
- Claude Code docs: after a gap longer than the TTL "the next request recomputes the full input and re-establishes the cache"; the cache is "effectively scoped to one machine and directory" (05-research-agents).
- Fireworks docs: prompt caching "only works within 1 replica"; cached prompts stay "for at least several minutes… it can be up to several hours" (05-research-providers).
- OpenAI docs: `prompt_cache_key` "does not pin requests to a machine or guarantee a cache read hit"; engines handle "roughly ~15 requests per minute per prefix" before overflow (05-research-caching).
- LMCache paper: "context truncation reduced prefix hit ratio from ~85% to ~45%" (05-research-kvcache, arXiv 2510.09665).

The pattern: the corpus is re-paid whenever a TTL expires, a request lands on the wrong replica, or the window fills. A cartridge is the bet that a corpus can be paid for once, as a file.

## 4. Value proposition and the proof-of-value benchmark

**Value proposition (three claims, all currently unproven).** (1) Memory: resident KV scales with compact length *t*, not corpus length *T*. (2) Beyond-window: a corpus larger than the model's context can be compiled into a prefix that fits. (3) Portability: a cartridge is a `.npz` keyed to exact model weights, loadable on any replica, immune to cache eviction.

**The benchmark (already pre-registered in `docs/BENCHMARK.md`).**
- Metric: token-F1 (and substring-EM) on LongBench single-doc QA (NarrativeQA, Qasper, MultiFieldQA), plus RULER multi-needle recall at 8k and 16k. Memory axis: bytes of KV held for the corpus, reported as compression ratio *T/t*.
- Setup: `unsloth/Llama-3.1-8B-Instruct` bf16 on A100-80GB (Modal), the same model/GPU as the 8B Attention-Matching runs. n ≥ 50 examples per LongBench subset, 6 seeds on RULER (matching `configs/harden-8b.yaml`). Harness: `benchmarks/frontier_bench.py` → `src/dexa/bench/frontier.py`, `datasets.py`, `qa_metrics.py`.
- Named baselines on one plot: `full_context` (ceiling), `rag` at k ∈ {1,3,8} (TF-IDF retriever in `src/dexa/bench/corpus.py`; a dense retriever is a follow-up), `heavy_hitter` (H2O), `snapkv`, `attention_matching` — all at matched ratios 4x/16x/50x/128x.
- Target numbers (verbatim from `docs/BENCHMARK.md`, at a fixed ratio in [16x, 50x]): cartridge F1 ≥ full_context − 2, AND ≥ best training-free method + 5, AND ≥ RAG + 3.
- Why a skeptical buyer would believe it: the baselines include the strongest repo-measured compactor (H2O at 128x multi-needle, 0.82 on 8B), thresholds were written before the run, the script and seeds are public, and the same harness produced the negative AM-vs-H2O result the repo published rather than hid.

**The make-or-break variable is the self-study data.** `docs/CARTRIDGES.md` attributes the 360M failure to distribution: generic prompts do not carry facts; corpus-span distillation "actively hurts QA." `src/dexa/cartridge/compiler.py` already implements corpus-conditioned Q&A generation (`_selfstudy_qa`) but with greedy decoding and `n_selfstudy=16` defaults. The experiment must use a capable teacher with sampling and hundreds to thousands of items — this has never been run.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| `dexa.cartridge.artifact.Cartridge` | custom (built, tested) | `[L, n_kv, t, d]` K/V + positions + `logical_length` + spec; `.npz` save/load; `to_compact_cache()` with zero bias |
| `dexa.cartridge.compiler.CartridgeCompiler` | custom (built; quality unproven) | warm-start from linspace downsample of corpus KV; corpus-conditioned Q&A self-study; Adam on K/V minimizing KL(teacher‖student) through the frozen model |
| Teacher / self-study generator | new (vLLM-served 8B, sampling) | fact-bearing Q&A at scale; the variable the whole result rests on |
| `dexa.engine.vllm_cartridge` layout helpers | custom (built, tested) | `cartridge_to_token_major`, `pack_token_major_into_blocks` (already reused by `vllm_connector.py:95`) |
| Cartridge loader (vLLM KVConnector V1) | new, on `DexaConnector` lifecycle | fill a request's prefix blocks from the cartridge; positions of query tokens rebased to *T*; `_PagedPrefixWriter.attach_to_runner` currently raises `RuntimeError` (`vllm_cartridge.py:488`) |
| vLLM 0.24 attention, scheduler, paged KV | OSS, unmodified | serving; no custom kernel because bias = 0 |
| Registry / versioning | new | cartridge id = hash(corpus, model weights hash, compiler config); tenant scoping |
| fp8 artifact option | custom, method proven for raw KV | halve bytes (`evals/modal_kv_interchange.py`) |
| Benchmark harness | custom (built) | `bench/corpus.py` (three-way + break-even), `bench/frontier.py`, LongBench/RULER loaders |
| STILL amortized compiler | custom (smoke-trained only) | v2 option: compile in a forward pass; not on the MVP path |

Where the differentiation lives: the compiler and the artifact. What is built *on* vLLM: everything at serve time — a cartridge is an ordinary prefix, so the stock kernels, scheduler, and prefix cache are untouched. What is *replaced*: nothing in the engine; the only in-engine code is a load-only connector.

```
 OFFLINE (compile once per corpus version)                ONLINE (serve)
 corpus ──► teacher prefill ──► Q&A self-study ──► KL-distill K/V ──► corpus@v3.cartridge
 (T tok)    (full context)      (sampled, n≈1k)   (frozen 8B, Adam)   [L,n_kv,t,d] + positions
                                                                            │
                                                                            ▼
                                            registry (model hash, tenant, fp8/bf16)
                                                                            │
 query ──► vLLM 0.24 (stock kernels) ◄── cartridge connector fills prefix blocks (t tokens)
           query positions start at T; decode attends t + query tokens
```

## 6. Evidence

### Proven
- Portable KV artifacts load into a separate vLLM process with prefill skipped and bit-identical output at TP=1 (OPT-125m/A10G; Qwen3-0.6B 8176/8192 tokens matched) — `cross-instance-resume`, `docs/RESULTS.md` 2026-07-09.
- `DexaConnector` matches all 10 vLLM 0.24.0 KVConnectorBase_V1 hook signatures — `connector-conformance`, `benchmarks/results/2026-07-09-vllm-connector-check.json`.
- fp8 (e4m3) and int8 KV at 0.5x bytes give 100% greedy agreement over 48 tokens; int4 diverges at token 14 — `kv-interchange-formats`.
- KV computed by one model is unusable by a differently fine-tuned model of the same architecture (base↔instruct diverges within 2–5 tokens) — `kv-interchange-weights`. Consequence: a cartridge is keyed to exact weights.
- Compiler mechanics: KL converges to ~0 on the CPU smoke; artifact round-trips; `test_corpus_bench` runs all three methods end-to-end on tiny-random Llama — `docs/CARTRIDGES.md` §Status; 03-build-inventory (136 passed / 18 skipped of 145; torch-gated cartridge tests skipped in that environment).

### Bounded / contradicted
- **Training-free compaction loses multi-fact fidelity at high ratios.** Llama-3.1-8B, 8k+16k multi-needle, 6 seeds: at 128x AM 0.53 vs H2O 0.82 vs SnapKV 0.86; AM wins only at ≥512x (0.55 vs 0.41) and 1024x (0.51 vs 0.33) — and all methods are near 0.5 there. Single-needle 8B at 128x: AM 0.96 / H2O 0.90 / SnapKV 0.90 — `attention-matching-vs-h2o`. This is the bar a *trained* cartridge must clear; it says nothing yet about cartridges themselves.
- **STILL is smoke-trained only**: KL decreases over ~12 steps on tiny-random Llama; identity-init reconstruction verified to ~1e-7; no quality number — `docs/RESULTS.md` §5.
- **The repo's cost model contradicts its own vLLM measurement.** `bench/corpus.py` prices per-query cost as decode time proportional to cache length (`_decode_seconds`). But `vllm-decode-pathology` measured n=1 decode/step flat at ~22 ms from 4k to 128k. At single stream, a cartridge's decode-time advantage over full context is therefore unsupported; the defensible per-query levers are memory occupancy (concurrency) and avoided prefill on cache miss, which the harness does not model.
- **Break-even is modeled, not measured** (arithmetic on `CostModel` defaults in `src/dexa/core/types.py`: 9,000 prefill tok/s, 120 decode tok/s, $1.80/hr, 320 KB/token; flagged as derived): T=16k, t=320 (50x), 256 self-study items, 200 steps → compile ≈ 3,231 GPU-s (≈54 min, ≈$1.62); break-even vs full-context ≈ 773 queries; vs TF-IDF RAG (k=3, 128-token chunks) ≈ 54,000 queries. The 320 KB/token constant is ~2.4x the measured 131 KB/token for Llama-8B bf16 (`docs/RESULTS.md` persist bench). Compile GPU-seconds on a real model: unmeasured.
- **In-engine long context is capped at 16k in the repo**: vLLM V1 crashed at ≥32k in three configs (`vllm-warmstart-prefix-cache`); the "beyond the window" claim has no in-engine measurement at any length.

### Unproven

| claim | experiment | est. cost |
|---|---|---|
| Cartridge QA quality meets `docs/BENCHMARK.md` thresholds at 16–50x on an 8B model | Section 4 benchmark with sampled corpus-conditioned Q&A (n_selfstudy 256–2,048, temp 0.8, 3 seeds) | ≈40 A100-hours ≈ $100 at Modal $2.50/hr |
| Cartridge holds multi-fact recall where H2O fails (128x multi-needle ≥ 0.82) | RULER multi-needle 8k/16k, 6 seeds, `configs/harden-8b.yaml` baselines | ≈10 A100-hours ≈ $25 |
| A corpus larger than the window compiles to a useful prefix | 64k–128k corpus (YaRN or native-128k model) into t ≤ 8k; LongBench-style QA | ≈15 A100-hours; blocked until the 16k result is positive |
| vLLM prefix injection is exact | HF `to_compact_cache()` path vs vLLM-injected prefix: ≥99% greedy agreement over 64 tokens on 20 prompts (method of `evals/modal_kv_interchange.py`) | ≈8 A100-hours + 1–2 engineer-weeks |
| Compile cost and real break-even | time compile on A100/H100; measure per-query TTFT and max concurrent cartridges at fixed HBM | ≈6 GPU-hours |
| fp8 cartridges keep quality | quantize trained K/V; rerun QA | ≈2 GPU-hours |
| Cartridge quality survives a corpus update (delta recompile) | edit 5% of corpus; warm-start from old cartridge; compare steps-to-threshold | ≈6 GPU-hours |

## 7. MVP and 6-week build plan

The MVP is: `dexa compile <corpus> --model <hf-id>` → artifact; `vllm serve … --kv-transfer-config CartridgeConnector` → an OpenAI-compatible endpoint accepting `cartridge: <id>`; plus the public benchmark.

- **Week 1 — the quality experiment, nothing else.** Stand up `benchmarks/frontier_bench.py` on Modal A100 with Llama-3.1-8B. Replace the greedy `_selfstudy_qa` teacher with a vLLM-served sampling teacher (new: `dexa/cartridge/selfstudy_vllm.py`). Run LongBench MultiFieldQA and RULER multi-needle at 16x and 50x with all baselines. Reuse: `src/dexa/cartridge/*`, `src/dexa/bench/{frontier,corpus,datasets,qa_metrics}.py`, `src/dexa/compaction/baselines.py`, `src/dexa/compaction/attention_matching.py`, `src/dexa/engine/hf_backend.py`, `configs/harden-8b.yaml`.
- **Week 2 — sweep and decide.** n_selfstudy {256, 1,024, 2,048} × steps {200, 1,000} × t at 16x/50x/128x; 3 seeds; NarrativeQA and Qasper. Gate G1 (Section 10) fires at end of week 2. If killed, the surviving assets (artifact format, benchmark harness, loader) fold into the KV-tiering candidates.
- **Week 3 — vLLM injection.** Implement the load-only connector on the proven V1 lifecycle: reuse `src/dexa/engine/vllm_connector.py` hooks (`get_num_new_matched_tokens`, `start_load_kv`, per-layer scatter) and `pack_token_major_into_blocks`; retire `_PagedPrefixWriter.attach_to_runner` (`vllm_cartridge.py:488`). TP=1, single attention backend, vLLM 0.24.0 pinned, matching the connector's known limits. Parity test (Section 6).
- **Week 4 — artifact and cost.** fp8 option via the `evals/modal_kv_interchange.py` quantizer; registry keyed by model-weights hash (the `kv-interchange-weights` result makes this mandatory); measure compile GPU-seconds and per-query TTFT vs full-context prompt cache and vs RAG on the same box; publish the measured break-even, replacing the modeled one.
- **Week 5 — endpoint.** Reuse `dexa_platform/gateway/app.py` (OpenAI-compatible proxy, tenant keys) with a `cartridge` request field; measure how many cartridges stay resident per A100 at fixed `gpu_memory_utilization` versus full-corpus KV; streaming must be added (gateway hard-sets `stream=False`, 03-build-inventory).
- **Week 6 — public proof.** Reproducible Modal script, the frontier plot, three design-partner corpora (one repo, one doc set, one KB) compiled and served; write-up in the style of `evals/RESULTS.md`, including negative results.

New code: sampling self-study generator; cartridge connector; registry; fp8 path; multi-cartridge residency test; corpus ingestion for repos (file ordering, tokenization budget). Not on the path: STILL, Attention-Matching warm start (the `docs/CARTRIDGES.md` "hybrid"), SGLang.

## 8. Pricing model

Facts that frame the price of *held context* today: Gemini explicit caches bill storage at $0.50–$4.50 per 1M tokens per hour with a 1-hour default TTL; Anthropic 1-hour caches cost 2x base input to write and 0.1x to read (0.025x on Fable 5.1); OpenAI retains up to 24h at no storage fee; DeepSeek bills cache hits at ~3% of miss price ($0.014 vs $0.44/M on V4-Flash) with disk storage free and eviction "within a few hours to a few days"; Tensormesh bills cached input at $0 and plans "30% of estimated savings" post-v1 (05-research-caching, 05-research-kvcache).

Options, all undecided:
1. **Compile fee + per-query.** A per-corpus compile priced at measured GPU-time plus margin; queries at a cached-input-style rate (the market's cached-read rates are 0.025x–0.5x of input). Simple; aligns with break-even.
2. **Cartridge-hours.** Price resident cartridges per hour, mirroring Gemini's token-hour storage but on *t* tokens instead of *T* — the memory claim monetized directly. Only meaningful if the memory ratio is proven.
3. **Open-source compiler + hosted serving**, the `README.md` Apache-2.0 posture; revenue from serving and registry.

Numbers that must be measured before any of these is set: compile GPU-seconds per 16k/64k corpus, cartridges resident per GPU, and the break-even under the buyer's actual query rate.

## 9. Competitive facts

| who | what they ship that is adjacent | what they do not ship per the research files | source |
|---|---|---|---|
| Frontier prompt caching (Anthropic, OpenAI, Google, DeepSeek) | prefix caching with TTLs of 5 min / 30 min–24 h / 1 h default / hours–days; Google bills storage per token-hour | no user-owned, portable, cross-provider context artifact; no context beyond the model window | 05-research-caching |
| Fireworks | automatic caching, `x-session-affinity`, per-replica cache | no cross-replica cache without affinity; no compression of context | 05-research-providers |
| LMCache / Tensormesh | KV storage across CPU/disk/S3, CacheBlend non-prefix reuse, $0 cached input; $24.5M raised | no trained/compressed KV; no beyond-window; a Q2-2026 roadmap item for a "token-dropping interface for sparse KV" | 05-research-kvcache |
| Morph Compact / FlashCompact | context compaction at 33,000 tok/s by line deletion, $0.20/$0.50 per M, default ratio 0.5 | no learned KV artifact; text-level, not KV-level; not beyond-window | 05-research-morph |
| Anthropic server-side compaction; Claude Code auto-compact | summarization at a 150k default trigger | lossy text summaries, not attention over the corpus | 05-research-agents |
| DeepSeek-OCR | optical context compression: 97% precision under 10x, ~60% at 20x | text-only corpora; not a KV artifact | 05-research-docai |
| SGLang HiCache / vLLM OffloadingConnector | tiered KV offload, LRU/ARC eviction | no compression, no compile step | 05-research-kvcache |

No vendor in the research files ships a trained, portable, compressed-KV context artifact; no vendor was found selling guaranteed session-pinned KV either (05-research-agents Unknowns).

## 10. Risks and pre-registered kill gates

| risk | measurement | number that kills | number that proceeds |
|---|---|---|---|
| G1: cartridge quality (the crux) | Section 4 benchmark, 8B, 3 seeds, end of week 2 | at 16x, cartridge F1 < full_context − 5 on ≥2 of 3 LongBench subsets, OR cartridge ≤ best of H2O/SnapKV/AM at 16x | all three `docs/BENCHMARK.md` thresholds met at some ratio in [16x, 50x]; in between = conditional (one more week, then decide) |
| G2: multi-fact fidelity | RULER multi-needle 8k/16k, 6 seeds | cartridge recall < 0.82 (H2O's 128x score) at 50x | ≥ 0.82 at 128x |
| G3: injection exactness | HF vs vLLM greedy agreement, 20 prompts × 64 tokens | < 95% | ≥ 99% |
| G4: compile economics | measured compile GPU-s and per-query cost on A100 | break-even vs full-context prompt cache > 10,000 queries per corpus version | ≤ 1,000 queries |
| G5: memory claim in-engine | cartridges resident per GPU vs full-corpus KV at fixed HBM, 16k corpora | < 4x more cartridges than full-KV corpora | ≥ 16x (the low end of the pre-registered ratio band) |
| G6: beyond-window | 128k corpus into ≤ 8k prefix, QA F1 | < RAG at the same query set | ≥ full-context on the 16k-truncated version of the same corpus |
| G7: vLLM API drift | connector conformance check on each vLLM release | two consecutive releases break the loader | conformance passes on 0.24 and the current release (0.28.0 per PyPI) |
| G8: model coverage | repeat G1 on a second family (Qwen2.5-7B) | thresholds fail on the second family | thresholds hold |

G4 and G5 thresholds are the product lead's proposals, not repo constants; the founder may reset them before the runs.

## 11. Founder decisions

1. **Run the experiment at all.** The repo's evidence is one negative small-scale attempt and a mechanism that works. Options: run the two-week experiment (≈$150 of Modal time); fold the artifact/loader into a KV-tiering candidate now; or shelve. Evidence that informs it: `cartridges-quality` (OPEN), the AM-vs-H2O multi-needle numbers.
2. **Which corpus type first.** Repo (coding agents; 60%-of-turns-searching claim), doc set, or per-tenant KB. Evidence needed: the discovery calls in Section 2.
3. **Open-source the compiler.** `README.md` says Apache-2.0 core; `dexa_platform/` says hosted PLG. The compiler is the differentiation; the loader is a commodity connector. Evidence: Tensormesh's OSS-core-plus-SaaS structure and $24.5M raised (05-research-kvcache).
4. **Model coverage.** A cartridge is per-exact-weights (`kv-interchange-weights`; LoRA reuse only to ~3% change). Supporting open models means one compile per model version; supporting frontier APIs is impossible. Options: one open family first; or a catalog.
5. **Position against RAG or beside it.** Under the repo's own cost model the cartridge beats RAG on cost only after ~54,000 queries; the pitch against RAG is quality and beyond-window, not cost. The founder decides whether that is the headline.
6. **Hosted vs BYOC.** The loader is a vLLM connector customers could run themselves; the compiler could be a service. Evidence: Fireworks/Baseten BYOC postures (05-research-providers).
7. **Engine breadth.** vLLM only (connector API "experimental") or also SGLang HiCache. Evidence: the API-drift history in 03-build-inventory.
8. **STILL (amortized compile).** Smoke-trained only; would remove the per-corpus training cost. Fund after G1, before, or never.
9. **Pricing model** (Section 8).
10. **The conditional band.** If G1 lands between kill and proceed — e.g. within 3 F1 of full-context but only tied with H2O — decide whether "portable and beyond-window at H2O quality" is a product.

## 12. Combinations

- **With the stateful-session candidate.** `README.md` §Status already frames this: keep recent context raw, compact only the cold tail. A cartridge is the cold-tail artifact; the session store and tiering policy carry the warm part. The bit-identical cross-instance resume applies unchanged to cartridges.
- **With the KV-tiering / voice candidates.** Per-customer system prompts in the voice traces are 300–1,600 tokens (02-evidence-ledger-voice) — too small for compression to matter; but per-tenant knowledge bases behind those agents are the cartridge case. Unmeasured.
- **With the agent-serving candidates.** A repo cartridge plus a WarpGrep-style search subagent attacks the same "60% of turns searching" claim from two sides; whether they are substitutes or complements is a G1-dependent question.
- **With Attention Matching / H2O.** `docs/CARTRIDGES.md` proposes an AM warm start to cut training steps. Given AM's 0.53 at 128x multi-needle, the warm start may inherit the fidelity loss; test only after G1 passes.
