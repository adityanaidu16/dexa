# Candidate G — Cartridges: the context compiler

## 1. Thesis

Compile a large static corpus (a repo, a doc set, a knowledge base) offline into a small trained KV artifact — a *cartridge* — that loads into stock vLLM as a precomputed prefix. Queries attend over the corpus without re-prefilling it, without holding full-corpus KV resident, and (the v2 claim) without the corpus fitting in the model window. The artifact is a file: versioned, shareable, movable across replicas, independent of any provider's cache TTL.

Honest state: the *systems* half (artifact format, compiler loop, layout helpers, connector lifecycle) is built and unit-tested on CPU/tiny models; the *quality* half is unproven and the one attempt (360M model, CPU) was negative (`cartridges-quality`, OPEN). `docs/CARTRIDGES.md`'s framing ("matches full-context quality at RAG cost, ~100x less memory") is on the superseded list in 04-conflicts.md and is not cited as fact here. Two things the first draft missed: (a) the repo's cartridge layout (compact keys at `linspace(0,T-1,t)`, query positions rebased to T) cannot be expressed through vLLM's KVConnector API, so "stock vLLM, no engine change" is conditional on retraining with contiguous positions; (b) the benchmark harness compiles one cartridge per example, so the draft's $100–150 experiment budget was unsupported. The plan is ~3 weeks to the quality answer under a pre-registered kill gate, then build on what survives.

## 2. Customer and workload

**Who.** Teams whose agents or applications ask many questions of the same large, slowly changing text: a coding-agent vendor with per-repo context; a product with per-tenant knowledge bases; a compliance/legal team over a regulatory corpus.

**Workload facts — and what they do not show.** Agent steps carry a median cached prefix of 126,180 tokens (Claude Code) / 115,584 (Codex) and append ~860 new tokens per step (TraceLab, arXiv 2606.30560); 96% of 219 production Claude Code requests reused ≥90% of input as a verbatim prefix (llm-d GLM-5.2 post); Anthropic server-side compaction triggers at a default 150,000 input tokens (05-research-agents). These describe a *growing per-session history*, not a static shared corpus; a cartridge does nothing for the per-step delta. On Alibaba Tongyi traces, 10% of KV blocks contribute 77% of reuses and cross-user reuse is "very low" (arXiv 2506.02634) — consistent with per-tenant artifacts, but measured on chat prefixes, not corpora. Morph's WarpGrep docs claim coding agents spend "60% of their turns searching for code" (05-research-morph; vendor claim).

**The regime cartridges target.** High reuse (hundreds to thousands of queries per corpus version), static content (a tagged release, a doc snapshot), corpora too large to keep resident per replica or larger than the window.

**Unmeasured.** No customer discovery exists; corpus-size, refresh-cadence and queries-per-corpus-day distributions are unknown for every segment. Experiment: ten discovery calls with coding-agent and KB-product teams collecting those three numbers. Cost: founder time.

## 3. The pain, in the customer's words

No cartridge customers exist; these are public statements from users of prompt caching, the incumbent a cartridge would replace. Quotes verbatim from the research files.

- OpenAI Codex issue #25604 (Azure): sessions past ~150K tokens "suffer total prompt-cache misses (cached_input_tokens = 0)"; one run re-billed 3.28M tokens, "71% of fresh input" (05-research-agents).
- Claude Code issue #46829 (community log audit, closed "not planned", no Anthropic primary statement): a 1h→5m TTL change computed as "17.1% overpayment ($949 on a Sonnet 4.6 sample)" across 119,866 calls (05-research-caching).
- Claude Code docs: after a gap longer than the TTL "the next request recomputes the full input and re-establishes the cache"; the cache is "effectively scoped to one machine and directory" (05-research-agents).
- Fireworks docs: caching "only works within 1 replica"; cached prompts stay "for at least several minutes… it can be up to several hours" (05-research-providers).
- OpenAI docs: `prompt_cache_key` "does not pin requests to a machine or guarantee a cache read hit"; engines handle "roughly ~15 requests per minute per prefix" before overflow (05-research-caching).
- LMCache paper: "context truncation reduced prefix hit ratio from ~85% to ~45%" (arXiv 2510.09665).

The pattern: the corpus is re-paid whenever a TTL expires, a request lands on the wrong replica, or the window fills. A cartridge is the bet that a corpus can be paid for once, as a file.

## 4. Value proposition and the proof-of-value benchmark

**Three claims, all unproven.** (1) Memory: resident KV scales with compact length *t*, not corpus length *T*. (2) Beyond-window: a corpus larger than the context compiles into a prefix that fits. (3) Portability: a `.npz` keyed to exact model weights, loadable on any replica; a cache miss costs a *t*-token reload, not a *T*-token prefill.

**The benchmark** (pre-registered in `docs/BENCHMARK.md`; revised here for metric consistency, cost and product shape).
- Metric: token-F1 and substring-EM on *generated* answers (`src/dexa/bench/qa_metrics.py`) for every method, including the training-free compactors. Not the `recall_frac` log-prob metric of the AM-vs-H2O runs, which is an affine rescaling that can exceed 1.0 and whose self-study reference queries can peek at the answer (01-evidence-ledger-dexa, `attention-matching-vs-h2o` caveats).
- Setup: `unsloth/Llama-3.1-8B-Instruct` bf16, A100-80GB on Modal, HFBackend via `benchmarks/frontier_bench.py` → `src/dexa/bench/frontier.py`. Because `run_frontier` compiles one cartridge per example per ratio, the corpus set is: ~10 LongBench Qasper/NarrativeQA documents with all their questions, plus 5 product-shaped corpora (2 repos, 2 doc sets, 1 KB) with ≥30 questions each, plus RULER multi-needle at 8k/16k (n=10, 3 seeds) as the multi-fact probe. Ratios 16x/50x/128x, 3 seeds.
- Baselines on one plot: `full_context` (ceiling); `rag` with the TF-IDF `BowRetriever` (256-token chunks) *and* BM25 + a dense retriever (new, ~2 days) at k∈{1,3,8}; `heavy_hitter` (H2O), `snapkv`, `attention_matching` at matched ratios, with `repeat_prefill` reference queries (the frontier default; no answer leak).
- Targets (verbatim from `docs/BENCHMARK.md`, at a ratio in [16x, 50x]): cartridge F1 ≥ full_context − 2, AND ≥ best training-free + 5, AND ≥ best RAG + 3.
- Why a skeptical buyer would believe it: thresholds were written before the run; every method is scored by the same generated-answer metric on the same items and seeds; the repo has already published a negative result from its own harness (AM 0.53 vs H2O 0.82 at 128x multi-needle) rather than hiding it; scripts, seeds and compiled artifacts are public. What it does *not* show: in-engine behaviour (G3), compile cost (G4), the memory claim under load (G5).

**The make-or-break variable is self-study data.** `docs/CARTRIDGES.md` attributes the 360M failure to distribution: generic prompts carry no facts; corpus-span distillation "actively hurts QA." `src/dexa/cartridge/compiler.py::_selfstudy_qa` is corpus-conditioned but greedy (`HFBackend.generate` defaults `greedy=True`) with `n_selfstudy=16`. The paper-style run — capable teacher, sampling, hundreds to thousands of items — has never been executed.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| `dexa.cartridge.artifact.Cartridge` | custom (built, tested) | `[L, n_kv, t, d]` K/V + positions + `logical_length`; `.npz` save/load; `to_compact_cache()` with zero bias |
| `dexa.cartridge.compiler.CartridgeCompiler` | custom (built; quality unproven; **layout change required**) | warm-start from downsampled corpus KV; corpus-conditioned Q&A self-study; Adam on K/V minimising KL(teacher‖student) through the frozen model. Today: compact keys at `linspace(0,T-1,t)`, query positions start at T (`tests/test_vllm_cartridge.py::test_query_positions_start_at_logical_length`). vLLM derives positions from sequence index and the KVConnector V1 API has no position hook (scheduler: `get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`, `request_finished`, `take_events`; worker: `start_load_kv`, `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`, `get_finished` — 05-research-kvcache). Fix: train at contiguous positions 0..t−1 with the query at t (paper-faithful), warm-starting via the exact `rope_rephase_keys` (`src/dexa/segment/selective.py`). Quality delta: unmeasured (G0). |
| Teacher / self-study generator | new (vLLM-served 8B, sampling) | fact-bearing Q&A at scale |
| `dexa.engine.vllm_cartridge` layout helpers | custom (built, tested) | `cartridge_to_token_major`, `pack_token_major_into_blocks` (reused by `vllm_connector.py:95`) |
| Cartridge loader (KVConnector V1, load-only) | new, on `DexaConnector` lifecycle | request = *t* placeholder tokens + query; connector reports the block-aligned prefix as externally computed, leaving ≥1 token for the engine (`vllm_connector.py:510`); after the first load vLLM's own prefix cache holds the blocks. Tenant isolation needs per-cartridge placeholder ids or vLLM `cache_salt`. `_PagedPrefixWriter.attach_to_runner` raises (`vllm_cartridge.py:488`) and is retired. |
| vLLM 0.24 attention, scheduler, paged KV | OSS, unmodified **iff G0 passes** | serving; no custom kernel because bias = 0 |
| Registry / versioning | new | id = hash(corpus, model-weights hash, compiler config); tenant scoping |
| fp8 artifact option | custom; method proven for *raw* KV only | halve bytes (`evals/modal_kv_interchange.py`); one 256-token passage, 48 greedy tokens, HF-level |
| Benchmark harness | custom (built; ran only on tiny-random Llama, CPU) | `bench/corpus.py`, `bench/frontier.py`, LongBench/RULER loaders |
| STILL amortised compiler | custom (smoke-trained only) | v2; not on the MVP path |

Differentiation lives in the compiler and the artifact. Built *on* vLLM: everything at serve time. Replaced: nothing in the engine, conditional on G0.

```
 OFFLINE (compile once per corpus version)                ONLINE (serve)
 corpus ──► teacher prefill ──► Q&A self-study ──► KL-distill K/V ──► corpus@v3.cartridge
 (T tok)    (full context)      (sampled, n≈1k)   (frozen 8B, Adam)   [L,n_kv,t,d], positions 0..t-1
                                                                            │
                                                                            ▼
                                            registry (model hash, tenant, fp8/bf16)
                                                                            │
 [t placeholders]+query ──► vLLM 0.24 ◄── load-only connector fills prefix blocks on miss
                            (stock kernels; prefix cache holds blocks after first load)
```

## 6. Evidence

### Proven
- Portable KV artifacts load into a separate vLLM process with prefill skipped and bit-identical output at TP=1 (OPT-125m/A10G; Qwen3-0.6B 8176/8192 tokens matched) — `cross-instance-resume`. Small models; single attention backend; single-step prefill only.
- `DexaConnector` matches all 10 vLLM 0.24.0 KVConnectorBase_V1 hook signatures — `connector-conformance`. Signature match only; a 3-arg constructor requirement surfaced later via live probe; the API is labelled experimental upstream.
- fp8 (e4m3) and int8 KV at 0.5x bytes: 100% greedy agreement over 48 tokens; int4 diverges at token 14 — `kv-interchange-formats`. One passage, 256-token context, HF bf16; trained K/V never quantised.
- KV from one model is unusable by a differently fine-tuned model of the same architecture (base↔instruct diverges within 2–5 tokens) — `kv-interchange-weights`. A cartridge is keyed to exact weights.
- Compiler mechanics: KL converges to ~0 on the 360M CPU smoke (`docs/CARTRIDGES.md` §Status) — while held-out QA did not beat no-context, so convergence is not quality; artifact round-trips; `tests/test_corpus_bench.py` runs full_context/rag/cartridge end-to-end on tiny-random Llama at `steps=2, n_selfstudy=2, t=8` (plumbing only). Test suite: 136 passed / 18 skipped of 145 in an environment without torch (03-build-inventory).

### Bounded / contradicted
- **Training-free compaction loses multi-fact fidelity at high ratios.** Llama-3.1-8B, 8k+16k multi-needle, 6 seeds: at 128x AM 0.53 vs H2O 0.82 vs SnapKV 0.86; AM wins only at ≥512x (0.55 vs 0.41) and 1024x (0.51 vs 0.33), where all are near 0.5. Single-needle 8B at 128x (4 seeds): AM 0.96 / H2O 0.90 / SnapKV 0.90 — `attention-matching-vs-h2o`. Metric is `recall_frac` with self-study references; not comparable to token-F1 without a rerun (G2).
- **STILL is smoke-trained only**: KL decreases over ~12 steps on tiny-random Llama (absolute KL ~5e-5 from near-uniform logits); identity-init reconstruction at t=T verified to ~1e-7 — `docs/RESULTS.md` §5.
- **The repo's cost model contradicts its own vLLM measurement.** `bench/corpus.py::_decode_seconds` prices decode ∝ cache length; `vllm-decode-pathology` measured n=1 decode/step flat at ~22 ms from 4k to 128k. At single stream the cartridge has no decode-time advantage; under batched decode the KV-bandwidth argument is unmeasured (G9).
- **Break-even is modelled, not measured** (arithmetic reproduced on `CostModel` defaults: 9,000 prefill tok/s, 120 decode tok/s, $1.80/hr, 320 KB/token): T=16k, t=320, 256 items, 200 steps → compile ≈ 3,231 GPU-s (self-study 1,274 s + training 1,957 s; ≈54 min, ≈$1.62); break-even vs full-context ≈ 773 queries and vs TF-IDF RAG (k=3, 128-token chunks) ≈ 54,000 queries — both resting on the contradicted decode term. 320 KB/token is ~2.4x the measured 131 KB/token for Llama-8B bf16 (`docs/RESULTS.md` persist bench; `configs/harden-8b.yaml` overrides to 262,144). The training term assumes a 344-token forward+backward costs a 9,000 tok/s prefill; HFBackend prefill on this model/GPU measured 3.3–4.2 s at 8k (~2–2.4k tok/s), so real compile is plausibly several times the proxy. Unmeasured.
- **In-engine long context.** The restore-vs-prefill runs are capped at 16k (Qwen2.5-7B + YaRN crashed vLLM V1 at ≥32k, `vllm-warmstart-prefix-cache`), but vLLM 0.24 did prefill Llama-3.1-8B at 128k in the repo (32.68 s eager, `context-scaling-branching`; SGLang at 130,944). The compiler's teacher is HFBackend, which OOMed at 128k on Qwen2.5-7B (`stateful-warm-session-hf`); a beyond-window compile needs a chunked or vLLM teacher.

### Unproven

| claim | experiment | est. cost |
|---|---|---|
| G0: contiguous-position cartridge matches the linspace/T layout on HF | 3 corpora × 2 layouts × 3 seeds at 50x; F1 delta | ≈6 compiles; unmeasured GPU-s |
| Cartridge QA quality meets `docs/BENCHMARK.md` thresholds at 16–50x on 8B | Section 4 benchmark, sampled Q&A (n_selfstudy 256; steps 200; 3 seeds) | ≈135 compiles + baselines; repo proxy ≈130 A100-h ≈ $325 at Modal $2.50/hr, plausibly 3–5x that (derived) |
| Self-study scale | n_selfstudy {1,024, 2,048} × steps {200, 1,000} on the 5 product corpora | ≈30 compiles at 4–40x unit cost |
| Multi-fact recall where training-free fails | RULER multi-needle 8k/16k, token-F1, all methods rerun in `frontier.py` | ≈120 compiles |
| Beyond-window | 64k corpus → t ≤ 4k with a chunked teacher; blocked until G1 positive | ≈15 A100-h + 1 week |
| vLLM injection exact | HF `to_compact_cache()` vs vLLM-injected prefix: ≥99% greedy agreement over 64 tokens on 20 prompts | ≈8 A100-h + 1–2 engineer-weeks |
| Compile cost and real break-even | time compile on A100/H100 day 1; per-query TTFT vs prompt cache, vs LMCache-CPU full-KV restore, vs RAG | ≈6 GPU-h |
| Batched decode saving | decode step at batch 32/64 with 16k vs 320-token shared prefix | ≈2 GPU-h |
| fp8 cartridges keep quality | quantise trained K/V; rerun QA | ≈2 GPU-h |
| Delta recompile | edit 5% of corpus; warm-start from old cartridge; steps-to-threshold | ≈6 GPU-h |

## 7. MVP and 6-week build plan

MVP: `dexa compile <corpus> --model <hf-id>` → artifact; `vllm serve … --kv-transfer-config CartridgeConnector` → OpenAI-compatible endpoint accepting `cartridge: <id>`; the public benchmark.

- **Week 1 — timing, G0, harness on GPU.** Day 1: time one compile (16k corpus, n=256, steps=200) on Modal A100 and re-base every budget below. Days 2–3: contiguous-position layout with `rope_rephase_keys` warm start; G0 on HF. Days 3–5: `frontier_bench.py` on GPU with Llama-3.1-8B (first real-model run of this harness; `datasets` dependency; expect debugging — the repo's KV-interchange eval took three iterations). Reuse: `src/dexa/cartridge/*`, `src/dexa/bench/{frontier,corpus,datasets,qa_metrics}.py`, `src/dexa/compaction/{baselines,attention_matching}.py`, `src/dexa/engine/hf_backend.py`, `src/dexa/segment/selective.py`.
- **Week 2 — teacher and first frontier.** vLLM-served sampling teacher (new `dexa/cartridge/selfstudy_vllm.py`); BM25/dense RAG baselines; product corpora authored; full run at 16x/50x, 3 seeds, all baselines under token-F1.
- **Week 3 — sweep and decide.** Self-study scale sweep on product corpora; RULER multi-needle rerun. **G1 fires end of week 3.** If killed, artifact format, harness and loader fold into the KV-tiering candidates.
- **Week 4 — vLLM injection.** Load-only connector on the proven V1 lifecycle (`vllm_connector.py` hooks, `pack_token_major_into_blocks`); placeholder-token prompts with `cache_salt`; TP=1, one attention backend, vLLM 0.24.0 pinned. Parity test (G3).
- **Week 5 — artifact, cost, memory.** fp8 path via the `modal_kv_interchange.py` quantiser; registry keyed by weights hash; measured compile GPU-s; per-query TTFT vs prompt cache and vs LMCache-CPU full-KV restore; cartridges resident per A100 at fixed `gpu_memory_utilization` (G4, G5, G9).
- **Week 6 — endpoint and proof.** `dexa_platform/gateway/app.py` with a `cartridge` field (streaming must be added: `app.py:127` hard-sets `stream=False`); reproducible Modal script; frontier plot; three design-partner corpora; write-up in the style of `evals/RESULTS.md`, negatives included.

New code: contiguous layout + re-phased warm start; sampling teacher; dense RAG baseline; product corpora; cartridge connector; registry; fp8 path; residency and batched-decode tests. Not on the path: STILL, AM warm start, SGLang.

## 8. Pricing model

Price of held context today: Gemini explicit caches bill storage at $0.50–$4.50 per 1M tokens per hour with a 1-hour default TTL; Anthropic 1-hour caches cost 2x base input to write and 0.1x to read (0.025x on Fable 5.1); OpenAI retains up to 24h with no listed storage fee; DeepSeek bills cache hits at ~3% of miss price ($0.014 vs $0.44/M on V4-Flash), disk storage free, eviction "within a few hours to a few days"; Tensormesh bills cached input at $0 and plans "30% of estimated savings" post-v1 (05-research-caching, 05-research-kvcache).

Options, all undecided: (1) compile fee at measured GPU-time plus margin, queries at a cached-input-style rate (market cached reads 0.025x–0.5x of input); (2) cartridge-hours — resident *t* tokens per hour, mirroring Gemini's token-hour storage, meaningful only if G5 passes; (3) open-source compiler + hosted serving/registry (`README.md` Apache-2.0 posture). Numbers required first: compile GPU-s per 16k/64k corpus, cartridges resident per GPU, break-even at the buyer's query rate.

## 9. Competitive facts

| who | what they ship that is adjacent | what they do not ship per the research files | source |
|---|---|---|---|
| Anthropic, OpenAI, Google, DeepSeek prompt caching | prefix caching with TTLs of 5 min / 30 min–24 h / 1 h default / hours–days; Google bills storage per token-hour | no user-owned portable context artifact; no compressed-KV product documented | 05-research-caching |
| Fireworks | automatic caching, `x-session-affinity`, per-replica cache | cross-replica cache without affinity; context compression not documented | 05-research-providers |
| LMCache / Tensormesh | KV across CPU/disk/S3, CacheBlend non-prefix reuse, $0 cached input; $24.5M raised | no trained/compressed KV; Q2-2026 roadmap lists "KV quantization" and a "token-dropping interface for sparse KV" | 05-research-kvcache |
| SGLang | HiCache L1/L2/L3; roadmap Q3 "HiSparse production readiness", session-aware RadixTree | no trained artifact documented | 05-research-kvcache |
| Morph Compact / FlashCompact | text-level compaction by line deletion, 33,000 tok/s, $0.20/$0.50 per M, default ratio 0.5 | KV-level or beyond-window compression | 05-research-morph |
| Anthropic server-side compaction; Claude Code auto-compact | summarisation at a 150k default trigger | attention over the corpus; summaries are lossy text | 05-research-agents |
| DeepSeek-OCR | optical context compression: 97% precision under 10x, ~60% at 20x | a KV artifact; text-only corpora | 05-research-docai |
| vLLM OffloadingConnector / KVBM / Mooncake | tiered offload with LRU/ARC (vLLM), presence/LFU filters (KVBM), leases and pins (Mooncake) | compression or a compile step | 05-research-kvcache |

None of the vendors in the research files is recorded as shipping a trained, portable, compressed-KV context artifact, and none was found selling guaranteed session-pinned KV (05-research-agents Unknowns). The research did not run a dedicated search for KV-compression products, so absence here is not evidence of absence.

## 10. Risks and pre-registered kill gates

| risk | measurement | number that kills | number that proceeds |
|---|---|---|---|
| G0: position layout | contiguous vs linspace/T on HF, 3 corpora × 3 seeds, 50x | contiguous F1 < linspace − 3 (then stock vLLM is false; an engine patch and version-pinned fork are required — founder decision) | within 1 F1 |
| G1: quality (the crux) | Section 4 benchmark, token-F1, 8B, 3 seeds, end of week 3 | at 16x, cartridge F1 < full − 5 on ≥2 of 3 corpus families, OR ≤ best training-free at 16x | all three `docs/BENCHMARK.md` thresholds at some ratio in [16x, 50x]; in between = one more week |
| G2: multi-fact fidelity | RULER multi-needle 8k/16k, all methods in `frontier.py`, same metric and seeds | cartridge below best training-free at 50x | ≥ best training-free at 128x |
| G3: injection exactness | HF vs vLLM greedy agreement, 20 prompts × 64 tokens | < 95% | ≥ 99% |
| G4: compile economics | measured compile GPU-s; per-query cost vs prompt cache | break-even vs full-context prompt cache > 10,000 queries/version, or compile > 4 GPU-h per 16k corpus | ≤ 1,000 queries and ≤ 1 GPU-h |
| G5: memory claim vs a tuned baseline | cartridges resident per A100 at fixed HBM vs full-KV bf16 and fp8; cold-corpus TTFT and p99 at concurrency {8,32,128} vs LMCache CPU-tier full-KV restore (10–17x vs prefill single-request at 4k–16k, `lmcache-offload-restore`; collapsed at N=300 on L40S in voice) | < 4x more resident corpora than fp8 full-KV, AND cartridge p99 not better than LMCache restore at 32-way | ≥ 16x resident and p99 better at 32-way |
| G6: beyond-window | 64k corpus → ≤ 4k prefix with chunked teacher; QA F1 | < best RAG on the same questions | ≥ full-context on the 16k-truncated corpus |
| G7: vLLM API drift | conformance on each release | two consecutive releases break the loader | passes on 0.24 and 0.28.0 (PyPI, 05-research-kvcache) |
| G8: model coverage | repeat G1 on Qwen2.5-7B | thresholds fail | thresholds hold |
| G9: batched decode | decode step at batch 64, 16k vs 320-token prefix | < 1.2x | ≥ 2x |
| G10: schedule | harness first real-model result | not by end of week 2 → G1 slips to week 4 and weeks 4–6 compress | by day 10 |

G4, G5, G9 thresholds are the product lead's proposals, not repo constants.

## 11. Founder decisions

1. **Run the experiment at all.** Evidence: one negative 360M attempt, a working mechanism, and the AM-vs-H2O numbers. Options: fund ~3 weeks and $1k–5k of Modal time (re-based after day-1 timing); fold artifact/loader into a KV-tiering candidate now; shelve.
2. **Benchmark shape.** LongBench-style one-doc-per-example (standard, but one compile per example) vs few corpora × many questions (product-shaped, cheaper, less standard). This revision chose the latter; the founder may reverse it.
3. **Layout.** If G0 fails: keep linspace/T and patch vLLM's model runner (a fork, version-pinned) or accept the contiguous-layout quality loss.
4. **Loader substrate.** `DexaConnector` (in-repo, TP=1, proven lifecycle), a backend plugin for vLLM's OffloadingConnector, or an LMCache/HiCache tier. Evidence: 05-research-kvcache API facts; 03-build-inventory connector limits.
5. **Which corpus type first.** Repo, doc set, or per-tenant KB. Evidence: discovery calls.
6. **Open-source the compiler.** `README.md` says Apache-2.0 core; `dexa_platform/` says hosted PLG. Evidence: Tensormesh's OSS-core-plus-SaaS structure and $24.5M raised.
7. **Model coverage.** Per-exact-weights (`kv-interchange-weights`; LoRA reuse ~3%, a one-line summary on a random-delta proxy). One open family first, or a catalog; frontier APIs impossible.
8. **Position against RAG or beside it.** Under the repo's contradicted cost model the cartridge beats RAG on cost only after ~54,000 queries; the RAG pitch is quality and beyond-window.
9. **Hosted vs BYOC.** Evidence: Fireworks/Baseten BYOC postures (05-research-providers).
10. **Engine breadth.** vLLM only (experimental connector API) or SGLang.
11. **STILL.** Fund after G1, before, or never.
12. **Pricing model** (Section 8).
13. **The conditional band.** If G1 lands within 3 F1 of full-context but only tied with H2O, whether "portable, beyond-window at H2O quality" is a product.
14. **First model and compute.** Llama-3.1-8B on Modal A100 was chosen for continuity with repo evidence; H100/newer models are untested.

## 12. Combinations

- **Stateful-session candidate.** `README.md` §Status frames it: keep recent context raw, compact only the cold tail. A cartridge is the cold-tail artifact; the bit-identical cross-instance resume applies unchanged.
- **KV-tiering / voice.** The voice traces' per-customer system prompts are synthetic, N(800,150) tokens clipped to [300,1,600] (02-evidence-ledger-voice) — too small for compression; the per-tenant knowledge bases behind such agents are the cartridge case. Unmeasured.
- **Agent-serving candidates.** A repo cartridge plus a WarpGrep-style search subagent attack the same "60% of turns searching" claim from two sides; substitutes or complements is G1-dependent.
- **Attention Matching / H2O.** `docs/CARTRIDGES.md` proposes an AM warm start; given AM's 0.53 at 128x multi-needle, test only after G1.
