# Evidence ledger — dexa repo

Provenance: /home/user/dexa has 50 commits, 2026-07-15 (3e57f9d) to 2026-09-02 (29df381, adds docs/FINDINGS.md only). docs/RESULTS.md, README.md, benchmarks/results/*.json and the compaction/persist/connector work (dated 2026-06-28 and 2026-07-09 inside the docs) were imported in the initial commit and have no per-experiment git history. All evals in evals/RESULTS.md ran on Modal (A100-80GB unless noted) with vLLM 0.24.0 pinned; SGLang and LMCache versions are unpinned in the scripts. Session attribution in commit trailers: Claude Opus 4.8. Cross-doc disagreements, stated explicitly: (1) docs/FINDINGS.md's meta-finding ('stateful sessions grew with scale, 10-34x, three reproductions') omits the docs/RESULTS.md 2026-07-09 results where raw-KV loading through the Dexa connector lost to vLLM re-prefill single-request (0.37x at 8B/8k) and under 16-24-way concurrency (P99 1.6-2.2x worse even with async loading; 'vLLM's re-prefill beats KV loading in every practical regime' until ~64k). (2) docs/INTEGRATION.md still states 'resume-from-state vs cold re-prefill (lossless, 14-25x faster, grows with length)' and 'connector pending validation on a real vLLM' - both superseded by docs/RESULTS.md (the 14-25x was SmolLM2/CPU; the GPU 8B result was 0.6-0.7x pre-fix, 3.7-5.5x post-fix vs HF prefill). (3) dexa_platform/README.md and docs/FINDINGS.md present '2-6x cheaper' as a measured advantage; it is a cost model with assumed rates (evals/stateful_cost_model.py). (4) docs/RESULTS.md records AM beating H2O at 128x on single-needle (SmolLM2 and 8B) but losing at 128x on 8B multi-needle (0.53 vs 0.82); the doc reconciles this as a regime crossover near 256-512x. (5) README.md's '3.7-5.5x faster than re-prefilling' uses HFBackend prefill as baseline; the same doc set measures vLLM prefill of 8k on the same model/GPU at 617 ms, versus which the connector resume was 0.37x. Numbers recorded only as one-line summaries with no run output in the repo: learned KV bridge (0%->6%), LoRA reuse (~3%), GPT-4o DocVQA 0.880 (GPT-4o-mini accuracy and measured $/1k pages not recorded), incremental recompute 4.6x, blob format 1.7x. Unmeasured claims in README.md 'The problem' section (up to ~40% of prefill compute redundant; ~90% slowdowns and O(N^2) blowups on mutation; 128K context ~40 GB KV) have no supporting experiment in the repo. Independent-benchmark survey (docs/research/2026-07-09-independent-benchmarks.md): 23 claims confirmed, 2 refuted - including the Mooncake '46x TTFT' figure, which the repo says not to cite; the planned Mooncake-trace head-to-head vs LMCache (docs/BENCHMARK_PLAN.md) was never run and README Phase-0 gate ('match or beat LMCache at 128K') is marked unmet. Build inventory as recorded (not re-verified here): core test suite '62 passed, 3 skipped' at docs/RESULTS.md time; cartridges '82 passing'; dexa_platform 15 tests (README) later '29 tests' (FINDINGS.md); serve/ has three Modal backends (cua_backend.py Qwen2.5-VL-7B, modal_doc_vlm_serve.py, vllm_lmcache_backend.py Qwen2.5-7B-Instruct served under the name 'dexa-cua-vlm'); edge/ Cloudflare scaffold typechecks but has never been deployed. Connector limits as recorded: TP=1 only, single attention backend, chunked prefill unsupported, whole-prompt keying (no block-hash keying), contention-aware loading shipped OFF.

## [PROVEN] compute-allocation-map — Test-time search (best-of-N / self-consistency) buys quality on verifiable reasoning tasks (code, math) but is flat on knowledge tasks.

- **Headline:** HumanEval pass 0.689 -> 0.902 (+21.3 pts) from N=1 to N=16; TriviaQA 0.680 -> 0.693 (+1.3 pts)
- **Numbers:** HumanEval N1/N4/N16 = 0.689/0.829/0.902, tokens 52k -> 858k | GSM8K 0.880/0.887/0.933, 33k -> 540k | TriviaQA 0.680/0.687/0.693, 0.9k -> 15k
- **Setup:** Llama-3.1-8B-Instruct, A100-80GB, vLLM 0.24.0, 150 problems/task (HumanEval all 164), SamplingParams n=N, temp 0.8 (0.0 at N=1), top_p 0.95, seed 0; GSM8K majority vote, HumanEval pass@k via unit tests, TriviaQA alias-match self-consistency. Undated in doc; predates repo history start 2026-07-15.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Compute-allocation eval'; /home/user/dexa/evals/modal_eval_compute.py
- **Caveats:** Single model. pass@k is an oracle selector (sampling ceiling). TriviaQA verifier noise. Doc labels it 'directional, not a leaderboard'. The +21 pts costs ~16x tokens naively.
- **Product relevance:** Supports: best-of-N with a unit-test verifier lifts 8B code pass ~21 pts. Does not support: any search benefit on knowledge QA.

## [PROVEN] verifier-early-stop — Verifier-guided early-stop reaches the same pass@16 as naive best-of-16 at a fraction of the tokens and wall-clock.

- **Headline:** 2.63x fewer decode tokens and 2.12x faster end-to-end at identical pass@16 (0.884 = 0.884), continuous scheduler
- **Numbers:** Post-hoc (modal_verifier_search): naive 860,830 tok pass@16 0.909 | early-stop B=4 302,349 tok (2.85x) | B=1 182,431 tok (4.72x) | avg samples used B=1 3.11/16. Live round engine (modal_verifier_engine): naive n=16 pass 0.896, 858,658 tok, 168.1 s gen wall, 27.0 s verify | engine B=4 pass 0.878, 330,174 tok (2.60x), 110.0 s (1.53x), 18.6 s verify. Continuous scheduler (modal_verifier_sched): naive 0.884 / 862,050 tok / 183.8 s | scheduler 0.884 / 327,607 tok (2.63x) / 86.6 s (2.12x).
- **Setup:** Llama-3.1-8B-Instruct, A100-80GB, vLLM 0.24.0, N=16, B=4, all 164 HumanEval, temp 0.8, top_p 0.95, max_tokens 640, seed 0; scheduler drives vLLM LLMEngine add_request/step/abort. Commits 3e57f9d and 354a3b7, 2026-07-15.
- **Source:** /home/user/dexa/evals/RESULTS.md sections 'Efficiency proof', 'Live engine', 'Continuous-batch scheduler'; /home/user/dexa/evals/modal_verifier_search.py, modal_verifier_engine.py, modal_verifier_sched.py
- **Caveats:** Verifier = full hidden test suite (oracle) in these three runs. Round-engine pass 0.878 vs 0.896 attributed to i.i.d. redraw noise. Later SGLang runs (agentic-value v1, contention v2) show ~1.0x GPU-seconds on an idle GPU and ~1.85x under load. docs/FINDINGS.md labels it 'buyer-replicable, not a moat'.
- **Product relevance:** Can claim pass@16 quality at ~2.6x fewer tokens / ~2.1x faster on HumanEval with an 8B model. Repo notes the same gain is largely capturable by round-based early-stop in client orchestration on any provider.

## [PROVEN] verifier-weak-robustness — The early-stop win holds on a second model and with a weak (visible-tests-only) verifier.

- **Headline:** 93-96% of oracle pass@16 kept at 3.0-3.5x lower cost with a first-3-asserts verifier, on two models
- **Numbers:** Llama-3.1-8B-Instruct: oracle pass@16 0.909, weak 0.848 (93%), visible FP rate 7.7%, oracle cost 2.85x, weak cost 3.04x | Qwen2.5-Coder-7B-Instruct: 0.963 / 0.921 (96%) / 3.8% / 3.45x / 3.52x
- **Setup:** N=16, all 164 HumanEval, A100-80GB, vLLM 0.24.0, temp 0.8, top_p 0.95, max_tokens 640, seed 0; weak verifier = first 3 asserts of check(); final quality scored on hidden suite. Commit 166b24b, 2026-07-24.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Robustness - 2nd model + a weak, deployable verifier'; /home/user/dexa/evals/modal_robustness.py
- **Caveats:** 'visible = first 3 asserts' is a proxy; HumanEval has no clean visible/hidden split. 4-8% of ceiling lost to visible false positives plus 6-10 problems/model with no visible pass within N.
- **Product relevance:** Supports 'keep >90% of the search gain at ~3x less cost with only example tests' on 7-8B code models; not 'lossless'.

## [FALSIFIED] context-scaling-branching — Prefill-once-then-branch (vLLM n=N) advantage over N independent re-prefills grows with context length.

- **Headline:** Speedup shrinks 3.03x at 4k -> 0.99x at 128k
- **Numbers:** L / prefill s / shared(n=8) s / naive(8 re-prefill) s / speedup: 4k 0.27/7.52/22.77/3.03x | 16k 1.35/13.82/32.16/2.33x | 32k 3.48/32.57/48.90/1.50x | 64k 10.06/91.76/101.41/1.11x | 128k 32.68/285.72/283.16/0.99x. Per-step 8-wide decode = 2.8x single at 4k, 94x at 128k.
- **Setup:** Llama-3.1-8B, A100-80GB, vLLM 0.24.0, N=8 branches, gen=128, prefix caching OFF, enforce_eager. Commit 11e6d10, 2026-07-25.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Long-context branching curve'; /home/user/dexa/evals/modal_context_scaling.py
- **Caveats:** One run; eager mode (doc argues eager penalizes naive baseline more, so shrink is robust). The ~5x 'ideal' is inferred from timing decomposition, not measured. 8 x 128k contexts do not fit 80 GB HBM.
- **Product relevance:** Cannot claim vLLM n=N shared-prefix branching saves wall time at long context; the prefill saving is consumed by decode.

## [PROVEN] vllm-decode-pathology — vLLM parallel sampling (n>1) over a long shared prefix has superlinear per-step decode cost.

- **Headline:** n=8 decode/step is 174x n=1 at 128k (ideal 8x); 4.6x at 4k; n=1 flat ~22 ms across lengths
- **Numbers:** decode/step n=1/2/4/8: 4k 21.7/26.4/34.5/100.9 ms (4.6x) | 32k 21.9/76.7/188/423 ms (19.3x) | 128k 21.9/554/1643/3812 ms (174x). n=1->n=2 at 128k = 25x jump.
- **Setup:** Llama-3.1-8B, A100-80GB, vLLM 0.24.0, gen=64, prefix caching off, eager; decode/step = (t_full - t_prefill)/gen. Commit 9595fbc, 2026-07-25.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Decode-pathology sweep'; /home/user/dexa/evals/modal_decode_pathology.py
- **Caveats:** vLLM 0.24 parallel sampling specifically. Eager mode compresses the ratio (understates). One run. Doc's own next-step: check SGLang before pitching (see sglang-gate).
- **Product relevance:** A measured vLLM 0.24 inefficiency; the SGLang gate shows it is engine-specific, so it does not support a 'new kernel/stack needed' claim.

## [FALSIFIED] sglang-gate — The shared-prefix / tree-native branched-decode white space is real across engines (SGLang also blows up).

- **Headline:** SGLang n=8 vs n=1 decode/step = 1.04x at 32k, 2.2x at 128k (vLLM: 19x / 174x)
- **Numbers:** SGLang decode/step n=1 / n=8: 32k 20.9 / 21.7 ms (1.04x) | 128k 21.1 / 46.7 ms (2.2x)
- **Setup:** Llama-3.1-8B, A100-80GB, SGLang (pip 'sglang[all]', version not pinned), cuda graph off, CONTEXT_LENS [4096, 32768, 130944], decode/step by differencing two generation lengths on a warm shared prefix; branches given unique final token. Commit 7657367, 2026-08-02.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'The SGLang gate'; /home/user/dexa/evals/modal_sglang_pathology.py
- **Caveats:** 4k n=1 point landed in timing noise (ratio artifact). SGLang version unpinned. First measurement was polluted by RadixAttention caching (fixed in 4c4bc9f).
- **Product relevance:** Cannot claim a cross-engine gap for branched decode over long shared prefixes; repo's conclusion: use SGLang rather than build a kernel.

## [PROVEN] kv-interchange-formats — KV cache can be stored/transported in a lower-precision numeric format and reloaded by the same weights without changing output.

- **Headline:** fp8 (e4m3) and int8 at 0.5x bytes: 100% greedy agreement over 48 tokens; int4 29% (diverges at token 14)
- **Numbers:** fp16 1.0x bytes, 100%, step-1 KL 0 | fp8 0.5x, 100%, KL 0.0003 | int8 0.5x, 100%, KL 0 | int4 0.25x, 29%, first divergence token 14, KL 0.0027
- **Setup:** unsloth/Llama-3.1-8B-Instruct, plain HF/torch bf16 (transformers 4.46.x), A100-80GB, single diverse 256-token context, greedy 48-token continuation, per-(token,head) symmetric quant. Commit fbe9a24, 2026-07-30.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'KV cache interchangeability - FORMAT axis'; /home/user/dexa/evals/modal_kv_interchange.py
- **Caveats:** One passage, 48-token horizon, greedy. Methodology note recorded: a tiled/repeated prompt produced a false 100% for every format and cross-model (induction copy artifact); took three iterations.
- **Product relevance:** Supports '2x smaller persisted/transported KV at fp8/int8 with unchanged greedy output' in this setup; int4 needs a smarter scheme.

## [FALSIFIED] kv-interchange-weights — KV computed by one model can be consumed by a differently fine-tuned model of the same architecture.

- **Headline:** base<->instruct KV injection diverges within 2-5 tokens despite K/V cosine 0.984/0.945
- **Numbers:** base gen <- instruct KV: 12.5% token agreement, first divergence token 5 | instruct gen <- base KV: 4.2%, token 2
- **Setup:** Llama-3.1-8B base + Instruct (unsloth), HF bf16, 256-token single-pass context, 48 greedy tokens, A100-80GB. Commit fbe9a24, 2026-07-30.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'WEIGHTS axis'; /home/user/dexa/evals/modal_kv_interchange.py
- **Caveats:** One model family, one passage, greedy decode.
- **Product relevance:** A KV store must be keyed to exact weights; cannot claim cache sharing across fine-tunes.

## [FALSIFIED] learned-kv-bridge — A per-(layer,head) linear map fitted by closed-form ridge regression can translate base-model KV into instruct-model space well enough to reuse it.

- **Headline:** Token agreement raw 0% -> bridged 6% (only a one-line summary is recorded)
- **Numbers:** docs/FINDINGS.md: 'learned bridge failed (0% -> 6%)'. Script docstring cites the raw baseline as ~4%. No reconstruction-cosine or per-layer table recorded anywhere in the repo.
- **Setup:** evals/modal_kv_bridge.py: unsloth/Meta-Llama-3.1-8B KV -> unsloth/Llama-3.1-8B-Instruct generation; ridge lam=0.1 per (layer,head) fitted on a calibration passage, tested on a held-out passage; k_gen=48 greedy; A100-80GB; transformers 4.46.3. Commit 918448f, 2026-07-30.
- **Source:** /home/user/dexa/docs/FINDINGS.md table row 'KV interchange (weights)'; method in /home/user/dexa/evals/modal_kv_bridge.py. Not recorded in /home/user/dexa/evals/RESULTS.md.
- **Caveats:** The only recorded output is the FINDINGS.md summary phrase; raw run output is not in the repo. Linear map only, no training loop, one calibration passage.
- **Product relevance:** Does not support a 'KV bridge across fine-tunes' product; the recorded result is a failure.

## [BOUNDED] lora-shared-base — A frozen base model's KV is reusable by a LoRA-adapted version ('prefill once with the base, serve many adapters').

- **Headline:** Base KV stays fully reusable only up to ~3% relative weight change (one-line summary only)
- **Numbers:** docs/FINDINGS.md: 'KV reusable only to ~3% weight change'. Sweep grid in script: s in {0, 0.01, 0.03, 0.10, 0.30}; per-strength KV cosine / agreement / first-divergence values are not recorded in the repo.
- **Setup:** evals/modal_kv_lora.py: unsloth/Meta-Llama-3.1-8B, rank-16 random low-rank delta on q/k/v/o/gate/up/down at relative strength s = ||delta||_F/||W||_F, k_gen=48 greedy, torch.manual_seed(0), A100-80GB. Commit a268367, 2026-08-02.
- **Source:** /home/user/dexa/docs/FINDINGS.md row 'LoRA over shared base'; /home/user/dexa/evals/modal_kv_lora.py
- **Caveats:** Random delta is a conservative proxy (script says a trained LoRA of equal norm should reuse at least as well). No results table in evals/RESULTS.md; only the summary number exists.
- **Product relevance:** FINDINGS.md verdict: 'too narrow to be a product'. Supports at most KV reuse for very small adapters.

## [FALSIFIED] agentic-serving-value-v1 — Agentic serving (shared prefix + B-way branch + verify + early-stop) beats generic serving by a margin that grows with context W and branching B.

- **Headline:** Up to 11.5x vs no-cache re-prefill, but only 1.00-1.02x vs a prefix-cached baseline
- **Numbers:** W/B: reprefill / cached / agentic GPU-seconds per turn: 4k/4 3.68/2.66/2.66 s (1.4x vs reprefill, 1.00x vs cached, 2x fewer decode tok) | 16k/8 13.27/2.68/2.66 (5.0x, 1.01x, 4x fewer) | 32k/8 30.65/2.70/2.66 (11.5x, 1.02x, 4x fewer)
- **Setup:** Llama-3.1-8B on SGLang, A100-80GB, dedicated-GPU wall-clock per turn, verification simulated by per-branch pass probability p. Commit 8f00ec0, 2026-08-02.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Agentic-serving value benchmark v1'; /home/user/dexa/evals/modal_agentic_value.py
- **Caveats:** Single idle GPU hides contention (addressed in contention-v2). Early-stop runs waves and can increase latency on hard turns. Verification simulated, not real.
- **Product relevance:** The large W x B-scaling win is RadixAttention prefix caching (free in SGLang) and cannot be claimed as differentiated; early-stop saves decode tokens (2-4x) but not GPU-seconds on an idle GPU.

## [BOUNDED] contention-v2-early-stop — Under multi-agent load, early-stop serving raises agent-turn throughput toward B/k.

- **Headline:** ~1.85x steady-state throughput at C=96 (generic 0.88 vs agentic 1.65 turns/s), not the 4x the token ratio predicts
- **Numbers:** concurrency C: generic / agentic turns/s / ratio: 1: 0.08/0.33/4.0x (cold-start artifact) | 8: 0.86/1.18/1.38x | 32: 0.93/1.57/1.69x | 96: 0.88/1.65/1.87x
- **Setup:** Llama-3.1-8B on SGLang, A100-80GB, W=4k, B=8, early-stop k=2, distinct per-agent contexts (no cross-agent sharing). Commit 09ab957, 2026-08-02.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Contention v2'; /home/user/dexa/evals/modal_contention.py
- **Caveats:** C=1 value is a JIT/cold-start spike; the script's auto-verdict compared it wrongly. Decode is bandwidth-bound so extra branches are partly free. Repo: a buyer captures most of the 1.85x with round-based early-stop in their own orchestration.
- **Product relevance:** Supports '~1.85x agent-turns/GPU under load' for this workload; repo's verdict: real efficiency, not a moat.

## [PROVEN] doc-vlm-frontier — Reducing the visual-token budget (via input resolution) on high-res documents cuts cost substantially at near-flat accuracy.

- **Headline:** 2.7x throughput at -1.0 pt accuracy (1024px, 0.925 vs 0.935); 4.1x at -5 pts (768px)
- **Numbers:** budget / acc / img/s / visual tokens / throughput / delta-acc: 1536px 0.935/11.4/2201/1.0x | 1024px 0.925/30.2/1017/2.7x/-1.0% | 768px 0.885/47.2/574/4.1x/-5.0% | 512px 0.735/78.6/277/6.9x/-20% | 384px 0.445/98.3/182/8.6x/-49%
- **Setup:** Qwen2.5-VL-7B on vLLM 0.24.0, A100-80GB, DocVQA validation 200 high-res pages, relaxed-match accuracy, warmup batch; v2 of the eval (v1 on ChartQA was invalid: images too small, cold-start in first budget). Commit 7237401, 2026-08-04.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Multimodal execution thesis'; /home/user/dexa/evals/modal_vlm_frontier.py
- **Caveats:** The lever is naive resize which a client can apply before any API; doc says this 'sizes the prize but isn't yet the moat'. 200 pages.
- **Product relevance:** Supports a configuration/economics claim for document AI (~2.7x at ~1 pt); not a defensible-kernel claim.

## [OPEN] vlm-moat-mrope — Content-aware in-model visual-token pruning beats naive resize at a matched token budget (an execution-owned moat).

- **Headline:** Blocked: Qwen2.5-VL mRoPE grid coupling breaks masked pruning ([3,1671] computed positions vs [3,881] masked)
- **Numbers:** No accuracy numbers produced. Planned FRACTIONS [0.5, 0.25] at 1280px long side, methods prune-norm / prune-query vs resize.
- **Setup:** evals/modal_vlm_moat.py, Qwen2.5-VL-7B via HF transformers 4.49.0, A100-80GB, DocVQA. Commit 7447e10, 2026-08-04.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Moat test - hit a real Qwen2.5-VL wall'; /home/user/dexa/evals/modal_vlm_moat.py
- **Caveats:** Architectural constraint (get_rope_index expects the full grid), not a bug; requires reimplementing position handling. Decision recorded: gate the sprint on market validation.
- **Product relevance:** No evidence either way for content-aware pruning; cannot claim a moat from it.

## [BOUNDED] gpt4o-docvqa-head-to-head — An open 7B VLM at a tuned visual-token budget matches GPT-4o on DocVQA at far lower cost per page.

- **Headline:** Qwen2.5-VL-7B 0.925 vs GPT-4o 0.880 on the same 200 DocVQA pages (measured); '~40x cheaper' is modeled from list prices
- **Numbers:** Accuracy: 0.925 (Qwen, 1024px, 1017 visual tok) vs 0.880 (GPT-4o). GPT-4o-mini accuracy: not recorded. Image tokens for a 1280x800 screenshot per pricing.py: Qwen ~322 (README says ~324), GPT-4o 1,105, GPT-4o-mini ~36,835 (README '~35,000'). Modeled $/M tokens: gpt-4o 2.50/10.00, gpt-4o-mini 0.15/0.60, claude-3-5-sonnet 3.00/15.00, dexa-cua-vlm 0.20/0.20 (assumed blended). Qwen cost printed as '$0.02-0.06 / 1k pages (rented A100 serving, not tokens)'.
- **Setup:** evals/modal_incumbent_docvqa.py: GPT-4o and GPT-4o-mini via OpenAI API, temperature 0, max_tokens 32, detail high, images capped to 2048 long side, same relaxed match, 200 lmms-lab/DocVQA validation pages. Commit c729976, 2026-08-04.
- **Source:** /home/user/dexa/docs/FINDINGS.md 'Multimodal / document VLM'; /home/user/dexa/dexa_platform/README.md 'Why it's cheaper'; /home/user/dexa/dexa_platform/gateway/pricing.py; /home/user/dexa/evals/modal_incumbent_docvqa.py
- **Caveats:** The measured GPT-4o/mini run output (measured $/1k pages, mini accuracy) is not recorded in the repo; only the 0.880 figure appears in FINDINGS.md and dexa_platform/README.md. The ~40x figure = (fewer image tokens) x (assumed $0.20/M rate vs $2.50/M), not a measured serving cost. Relaxed-match scoring; 200 pages; commit-message pricing 'early 2026 list prices'.
- **Product relevance:** Supports an accuracy-parity claim on DocVQA (200 pages, relaxed match). A '~40x cheaper' claim rests on the pricing assumptions in pricing.py.

## [PROVEN] cua-screen-redundancy — Computer-use agent screenshots are mostly redundant frame-to-frame, so encoding only changed patches would cut visual tokens ~7x.

- **Headline:** ~86% of 28px patches unchanged per action on average (13.7% changed) -> 7.3x headroom; 15-40x on typing/clicking actions
- **Numbers:** type / edit field / pick dropdown 2.5-6% changed | select a table row ~28% | full page navigation 22-46% | average per action 13.7%
- **Setup:** evals/agent_redundancy/bench.py: Playwright Chromium driving a synthetic CRM web app (app.html) through a 15-step task; 28x28 patches, mean abs luma diff threshold 6. Commit 9ed36e2, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Computer-use agent screen redundancy'; /home/user/dexa/evals/agent_redundancy/bench.py
- **Caveats:** Synthetic-but-real-browser task, not OSWorld/WebArena trajectories. Measures the prize (patch redundancy), not a realized win.
- **Product relevance:** Supports a headroom statement only; realized reuse is bounded by the delta-viability and tolerance entries (~2-2.3x).

## [BOUNDED] delta-perception-viability — Unchanged screen regions' vision-encoder embeddings stay stable across consecutive frames, so their computation can be reused.

- **Headline:** Vision-token shift is 3.39x the pixel change (52.8% of tokens shift for 15.6% pixel change) -> naive reuse ~2x, not 7x
- **Numbers:** small type/edit actions: 2.5-6% pixels -> 15-44% tokens shifted (cos<0.98) | full-page nav: 45-46% -> 93-95% | average 15.6% -> 52.8%
- **Setup:** Qwen2.5-VL vision encoder (HF transformers 4.49.0), consecutive Playwright frames of the CRM app, per-token cosine aligned by grid position, Modal GPU. Commit 4ce46cf, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Delta-perception viability'; /home/user/dexa/evals/agent_redundancy/modal_delta_viability.py
- **Caveats:** Threshold-sensitive (cos<0.98). Small-action reuse still 63-85%. Downstream grounding-accuracy tolerance not measured.
- **Product relevance:** A delta-encoding perception engine can claim ~2x on common actions, not 7x.

## [FALSIFIED] delta-tolerance-sweep — Much of the embedding spill is soft (cos 0.98-0.999) and reusable with tolerance, rescuing a ~5x multiple.

- **Headline:** Median cosine of moved tokens 0.83; capturable ceiling 2.34x at cos>=0.98, ~3.1x at >=0.95, ~3.9x at >=0.90 - never near 7x
- **Numbers:** moved-token bands: 0.95-0.98 24% | 0.90-0.95 15% | 0.80-0.90 14% | 0.50-0.80 27% | <0.50 20% (~47% below 0.80). n=8 small-action frames.
- **Setup:** Same frames as the viability run re-scored across a cosine-tolerance ladder; Qwen2.5-VL encoder. Commit a4e1058, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Tolerance sweep'; /home/user/dexa/evals/agent_redundancy/modal_delta_tolerance.py
- **Caveats:** n=8 frames. Grounding-accuracy test not run; the 0.95-0.98 band (24%) might later be cleared by such a test.
- **Product relevance:** Honest capturable ceiling ~2.3x (maybe ~3x); do not claim 5x.

## [PROVEN] stateful-warm-session-hf — Restoring a saved KV cache from pinned CPU (or NVMe) is faster than re-prefilling the context, and the gap widens with context length.

- **Headline:** restore(CPU) vs re-prefill: 11.8x at 4k -> 28.9x at 64k (HF tensor copies, Qwen2.5-7B)
- **Numbers:** ctx / KV / re-prefill / restore CPU / speedup / restore NVMe speedup: 4k 0.23 GB 296 ms 25 ms 11.8x 1.2x | 16k 0.94 GB 1,321 ms 115 ms 11.5x 3.4x | 32k 1.88 GB 3,167 ms 182 ms 17.4x 4.9x | 64k 3.76 GB 8,567 ms 297 ms 28.9x 7.0x. NVMe restore ms (cost-model DATA): 246/390/646/1228. Decode/step 39-108 ms in both regimes. Projection: 64k x 50 turns 428.4 s vs 23.4 s (18.3x); 32k x 50 158.4 s vs 12.3 s (12.9x). 128k OOM (HF non-paged artifact).
- **Setup:** Qwen/Qwen2.5-7B-Instruct bf16, sdpa attention, HF transformers 4.49.0, A100-80GB, evals/modal_stateful_session.py. Commit b83eec9, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Stateful warm-session vs re-prefill'; /home/user/dexa/docs/STATEFUL_SESSIONS.md; /home/user/dexa/evals/modal_stateful_session.py
- **Caveats:** HF-level physics, not vLLM paged KV. The HF prefill baseline is far slower than vLLM prefill (docs/RESULTS.md: vLLM 8B/8k prefill 617 ms vs HFBackend Llama-8B/8k 3.3-4.2 s), so ratios vs an optimized engine are smaller. NVMe tier uses naive torch.load. Single request only; no concurrency.
- **Product relevance:** Supports 'restore is bandwidth-bound and beats HF re-prefill 10-30x'; the vLLM-based entries are what a serving product would face.

## [PROVEN] vllm-warmstart-prefix-cache — Inside vLLM's real paged-KV engine, a prefix-cache hit (KV reuse) beats cold prefill, growing with context.

- **Headline:** cold vs warm TTFT ~12x at 4k (280 -> 24 ms), ~25-34x at 16k (1,250 -> 45 ms); >=32k crashed the engine
- **Numbers:** 4k ~280 ms -> ~24 ms (~12x) | 16k ~1,250 ms -> ~45 ms (34.4x / 27.9x / 25.0x over 3 runs) | 32k-64k: vLLM V1 EngineCore subprocess died across 3 configs (max_model_len 70k-133k, gpu-util 0.85-0.92, chunked prefill on/off)
- **Setup:** Qwen2.5-7B-Instruct, vLLM 0.24.0 V1, A100-80GB, prefix caching ON, YaRN factor 4, chunked prefill max_num_batched_tokens 8192, gpu_memory_utilization 0.85, enforce_eager False. Commit de61a36, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Production reproduction on vLLM paged KV'; /home/user/dexa/evals/modal_vllm_warmstart.py
- **Caveats:** The warm path is an in-GPU cache hit with no transfer - not an offload/restore. Confirmed only to 16k in-engine.
- **Product relevance:** Confirms vLLM prefix caching's value; per the agentic-value entry this is the mechanism vLLM/SGLang already ship ('table stakes').

## [PROVEN] lmcache-offload-restore — KV evicted from GPU to CPU via LMCache and restored on resume beats cold prefill in a real vLLM stack (prefix cache off).

- **Headline:** 10.0x at 4k (301 -> 30 ms), 17.1x at 16k (1,320 -> 77 ms); LMCache retrieved 16,384 tokens from CPU in 39.8 ms at 22.0 GB/s
- **Numbers:** 4k cold 301 ms / restore 30 ms (10.0x) | 16k cold 1,320 ms / restore 77 ms (17.1x) | 16k KV ~0.875 GB moved CPU->GPU in ~40 ms
- **Setup:** Qwen2.5-7B-Instruct, vLLM 0.24.0 V1 + LMCacheConnectorV1 (kv_both, lmcache version unpinned), vLLM prefix caching OFF, LMCACHE_LOCAL_CPU=True, 20 GB budget, chunk 256, max_model_len 20000, gpu util 0.80, A100-80GB. Commit 002f1b9, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Offload -> restore through vLLM + LMCache'; /home/user/dexa/evals/modal_lmcache_restore.py
- **Caveats:** Only 4k and 16k; single request, no concurrency; NVMe/disk tier not tested; LMCache emits a benign 'Pin count negative' warning. The restore mechanism is LMCache's; docs/STATEFUL_SESSIONS.md says the delta is the tiering policy, not the mechanism.
- **Product relevance:** Supports 'LMCache CPU offload/restore is ~10-17x faster than re-prefill at 4k-16k single-request'; does not support under-load or NVMe claims.

## [OPEN] residency-cost-model — Given tiering (HBM -> RAM -> NVMe -> drop), a stateful session is cheaper in dollars than re-prefill on long-context, idle-gapped sessions.

- **Headline:** Modeled: 6.0x / 4.9x / 2.0x cheaper at 2 / 15 / 120 min idle (64k, 50 turns); break-even idle at 64k: HBM ~2 min, RAM ~11 min, NVMe ~4.9 hr
- **Numbers:** Latency win 28.8x at 64k (0.3 s vs 8.6 s), 11-29x across sizes | break-even 4k: HBM 1 min, RAM 6 min, NVMe 33 min; 64k: 2 min, 11 min, 4.9 hr | assumed rates GPU $1.80/hr, RAM $0.006/GB-hr, NVMe $0.0002/GB-hr, 60 GB usable HBM; fp8 KV ~doubles break-evens
- **Setup:** evals/stateful_cost_model.py, pure Python over the HF-measured DATA (Qwen2.5-7B prefill/restore ms) - no new measurement. Same constants encoded in dexa_platform/sessions/tiering.py. Commit ef9ba6a, 2026-08-05.
- **Source:** /home/user/dexa/evals/RESULTS.md section 'Residency cost model'; /home/user/dexa/docs/STATEFUL_SESSIONS.md 'Proven vs. not proven' item 4 and 'Not yet proven' item 3; /home/user/dexa/evals/stateful_cost_model.py; /home/user/dexa/dexa_platform/sessions/tiering.py
- **Caveats:** Modeled, not measured; STATEFUL_SESSIONS.md lists live unit-economics validation as not yet proven. Uses HF prefill times (slower than vLLM), which inflates the re-prefill cost side. dexa_platform/README.md presents '2-6x cheaper' under 'Measured advantage' alongside the measured latency numbers.
- **Product relevance:** Can be presented as a cost model with stated assumptions; cannot be claimed as measured $ savings.

## [PROVEN] stateful-session-service-live — The session service end-to-end (HTTP -> vLLM+LMCache on Modal) restores a session warm on turn 2 faster than the cold turn 1.

- **Headline:** 12k-token session: turn 1 cold ~5.4 s, turn 2 warm ~1.5 s (3.7x end-to-end)
- **Numbers:** 5.4 s -> 1.5 s (3.7x); doc notes the isolated in-engine restore win is ~11x and HTTP + decode overhead compresses the end-to-end ratio
- **Setup:** dexa_platform/sessions/service.py against serve/vllm_lmcache_backend.py: Qwen2.5-7B-Instruct served under the name 'dexa-cua-vlm', vLLM 0.24.0 + LMCache CPU tier (40 GB, chunk 256), prefix caching off, max_model_len 20000, gpu util 0.80, A100-80GB on Modal. Commit c77e8cd, 2026-08-05.
- **Source:** /home/user/dexa/dexa_platform/README.md 'Stateful sessions'; /home/user/dexa/docs/FINDINGS.md 'What got built'; /home/user/dexa/serve/vllm_lmcache_backend.py
- **Caveats:** Single demo run at one context size; tiering policy is advisory (LMCache LRU still governs eviction); NVMe tier not driven; Cloudflare edge port typechecks but has not been run in CF.
- **Product relevance:** A working demo of turn-2 warm restore through a deployed stack; not a benchmark across sizes or under load.

## [CONTRADICTED] raw-kv-resume-vs-reprefill-gpu-arc — Resuming a persisted raw-KV session state on a GPU is faster than cold re-prefill (Dexa HFBackend persist bench, Llama-3.1-8B).

- **Headline:** 0.6-0.7x (resume slower) at 16k-64k before the bf16 load fix -> 3.7-5.5x faster after the fix, both vs HF prefill; but 0.37x vs vLLM prefill at 8B/8k (see connector entries)
- **Numbers:** Pre-fix (2026-07-09-a100-8b-persist.json): 8k cold 4412 / resume 3174 ms (1.39x, load 2302 ms) | 16k 4593 / 6464 (0.71x, load 4766) | 32k 9939 / 14338 (0.69x, load 11381) | 64k 20188 / 31428 (0.64x, load 25265; ~330 MB/s, degrading 456->332 MB/s). Post-fix (2026-07-09-a100-8b-persist-native-load.json): 8k 5816 / 1061 ms (5.48x, load 2.2 ms) | 16k 8665 / 2348 (3.69x, 4.2 ms) | 32k 17941 / 4765 (3.77x, 8.0 ms) | 64k 42221 / 9822 (4.30x, 14.2 ms). State 131 KB/token bf16 = 1.0 / 2.1 / 4.2 / 8.4 GB. identical_output true at every length in both runs. Earlier README/INTEGRATION claim of '14-25x faster' was SmolLM2 on CPU.
- **Setup:** unsloth/Llama-3.1-8B-Instruct bf16, HFBackend persist bench (benchmarks/persist_demo.py via scripts/modal_scale_and_connector.py --only persist), NVIDIA A100-SXM4-80GB, torch 2.13.0+cu130, 2026-07-09. Post-fix box ~1.7x slower in absolute terms (cold prefill 13.6 s -> 23.5 s at 64k).
- **Source:** /home/user/dexa/docs/RESULTS.md 'Update (2026-07-09): real 8B on A100 - falsified' and 'Update (2026-07-09, later): load-path fix flips resume to a GPU win'; /home/user/dexa/benchmarks/results/2026-07-09-a100-8b-persist.json and 2026-07-09-a100-8b-persist-native-load.json; /home/user/dexa/README.md status paragraph
- **Caveats:** Baseline is HFBackend prefill (8k = 3.3-4.2 s), not vLLM prefill (617 ms at 8k, same model/GPU, per docs/RESULTS.md). Post-fix load_ms of 2-14 ms reflects lazy mmap; bytes are actually moved inside resume_ms. Compare speedups within a run only. docs/INTEGRATION.md still states '14-25x faster, grows with length' and calls the connector 'pending validation'.
- **Product relevance:** Supports 'lossless, portable resume' and a ~4-5x speedup relative to an HF forward pass; does not support a TTFT win against vLLM re-prefill at <=32k.

## [FALSIFIED] connector-session-resume-single-request — Resuming a full session through the Dexa vLLM connector on a cold instance beats vanilla vLLM re-prefill (single request).

- **Headline:** Llama-3.1-8B/A100/8k: optimized resume 1681 ms vs vLLM cold re-prefill 617 ms (0.37x); Qwen3-0.6B/8k: 4721 -> 1073 ms after load optimization vs 273-311 ms cold (0.06x -> 0.29x)
- **Numbers:** Qwen3-0.6B, 8192 ctx, A10G: cold 273 ms, resume 4721 ms (8176 tokens matched, fp32 load of ~1.9 GB) | after vectorized index_select/scatter + keep_native: 1073 ms (4.4x faster) vs 311 ms cold, next-token correct | Llama-3.1-8B A100 8k: cold 617 ms vs resume 1681 ms (0.37x), next-token correct
- **Setup:** scripts/modal_bench_resume.py, vLLM 0.24.0, three Modal containers sharing a persistent Volume store, single-step prefill forced. 2026-07-09.
- **Source:** /home/user/dexa/docs/RESULTS.md 'Update (2026-07-09, session-resume benchmark)' and its two follow-ups
- **Caveats:** Connector loads .npz/blob from a Modal Volume (not pinned CPU RAM as LMCache does). Repo estimates the crossover at ~64k+ tokens, a regime that needs chunked-prefill support to benchmark. TP=1 only.
- **Product relevance:** Cannot claim TTFT improvement vs vLLM re-prefill at 8k-32k with this connector; value stated as portability/correctness.

## [FALSIFIED] connector-under-concurrency-sync-async — Under GPU contention, loading KV beats re-prefill because prefill queues while loads use idle I/O (contention-aware policy).

- **Headline:** Sync load P99 TTFT 18553 ms vs vanilla 5505 ms (3.4x worse); async 8930 vs 5596 ms; at 8192-tok x16 async 20296 vs 9379 ms; adaptive matches vanilla (5603 / 9371 ms)
- **Numbers:** 24 x 3072-token sessions, sync: vanilla p50/p99/mean 3083/5505/2965 ms | dexa_always 18548/18553/18546 | dexa_adaptive(contention-aware) 15885/15907/14347. Async re-run (same workload): vanilla p99 5596 | dexa_always(async) 8930 (2.1x better than sync) | dexa_adaptive 5603; single-request resume 0.29x -> 0.99x. 8192-token x 16 concurrent: vanilla 5359/9379/5068 | dexa_always(async) 15145/20296/12927 | dexa_adaptive 5350/9371/5061.
- **Setup:** scripts/modal_bench_contention.py, Llama-3.1-8B-Instruct, A100-80GB, vLLM 0.24.0, sessions pre-saved sequentially then replayed concurrently with streaming TTFT. 2026-07-09.
- **Source:** /home/user/dexa/docs/RESULTS.md top section 'Update (2026-07-09, contention benchmark)' incl. 'Follow-up - async loading built' and 'Crossover measurement'; /home/user/dexa/docs/CONNECTOR_COMPLETION.md 'Adaptive load-vs-recompute'
- **Caveats:** Dexa connector load path (Volume disk -> host -> GPU), not LMCache. Contention-aware policy loaded the later requests once its EMA ramped (the wrong call); shipped OFF by default. Repo conclusion: 'for re-prefillable KV, vLLM's re-prefill beats KV loading in every practical regime' until ~64k.
- **Product relevance:** Supports only 'the adaptive policy is never worse than baseline'; contradicts any under-load TTFT claim for raw-KV loading via this connector.

## [CONTRADICTED] restore-vs-reprefill-cross-repo-contradiction — General claim: restoring persisted KV beats re-prefilling (the repo holds results pointing both ways).

- **Headline:** 10-34x faster (Qwen2.5-7B; HF copies / vLLM prefix cache / LMCache CPU; single request; Aug 2026) vs 0.37x single-request and 1.6-2.2x worse P99 under concurrency (Llama-3.1-8B; Dexa connector; Jul 2026)
- **Numbers:** Side A: HF restore 11.8-28.9x | vLLM prefix-cache hit 12x / 25-34x | LMCache CPU restore 10.0x / 17.1x (4k/16k). Side B: Dexa connector 8B/8k 1681 ms vs 617 ms cold | 24x3072 async p99 8930 vs 5596 ms | 16x8192 p99 20296 vs 9379 ms. Per-byte: LMCache 0.875 GB in 39.8 ms (22 GB/s) vs Dexa connector 8B/8k ~1.7 GB in ~1.68 s (~1 GB/s).
- **Setup:** See entries stateful-warm-session-hf, vllm-warmstart-prefix-cache, lmcache-offload-restore (side A) and connector-session-resume-single-request, connector-under-concurrency-sync-async (side B).
- **Source:** /home/user/dexa/evals/RESULTS.md stateful sections vs /home/user/dexa/docs/RESULTS.md 2026-07-09 updates; /home/user/dexa/docs/FINDINGS.md cites only side A; /home/user/dexa/docs/STATEFUL_SESSIONS.md 'Not yet proven' item 2 (concurrent multi-session residency under load)
- **Caveats:** Confounds: different models (Qwen2.5-7B vs Llama-3.1-8B), store media (pinned CPU RAM vs Modal Volume disk), load implementations (LMCache vs Dexa connector), and regime (single request vs 16-24 concurrent). No run measures LMCache restore under concurrency. docs/FINDINGS.md's meta-finding does not mention the docs/RESULTS.md contention/crossover results.
- **Product relevance:** Any resume-speed claim must specify store tier, load implementation, model, and concurrency; the repo has evidence for both directions.

## [FALSIFIED] vllm-bench-serve-prefix-repetition — The Dexa connector is competitive with vanilla vLLM / prefix caching on an independent prefix-sharing serving benchmark.

- **Headline:** Dexa mean TTFT 3355 ms at 9.8 req/s vs no-cache baseline 610 ms / 52.3 req/s vs vLLM prefix cache 183 ms / 99.1 req/s (Dexa ~5.5x slower than no-cache)
- **Numbers:** prefixcache: 99.1 req/s, TTFT mean 183 / median 180 / P99 272 ms | baseline (no cache): 52.3 req/s, 610 / 601 / 841 ms | dexa: 9.8 req/s, 3355 / 3385 / 5854 ms. Zero connector hits: whole-prompt keying, 30+ distinct saves of ~160 MB each.
- **Setup:** vllm bench serve --dataset-name prefix_repetition, OPT-125m, A10G, 1024-token prefix, 60 prompts, vLLM 0.24.0, scripts/modal_bench_serve.py. 2026-07-09.
- **Source:** /home/user/dexa/docs/RESULTS.md 'Update (2026-07-09, independent benchmark)'; /home/user/dexa/docs/BENCHMARK_PLAN.md
- **Caveats:** Tiny model. Workload is prefix sharing (LMCache's domain) which the connector's whole-prompt exact-match keying does not serve; unconditional save on every request. The BENCHMARK_PLAN Mooncake-trace head-to-head vs LMCache was not run.
- **Product relevance:** Cannot claim parity with LMCache or vLLM prefix caching on prefix-sharing traffic; the planned independent head-to-head is unrun.

## [PROVEN] cross-instance-resume — KV saved by one vLLM process can be loaded by a separate vLLM process with prefill skipped and bit-identical output.

- **Headline:** Two Modal containers (pids 36 vs 216) sharing a Volume: A_saved=True, B_saw_stored_KV=True, identical_output=True; 8176/8192 tokens matched at 8k
- **Numbers:** Cross-request (one server): req 1 saved T=43; req 2 store HIT 32 external tokens (2 full blocks), re-prefilled remaining 11; identical_output=True. Cross-instance: 2 blocks loaded, identical output. Resume bench: 8176 tokens matched (block-aligned) of 8192.
- **Setup:** OPT-125m, A10G, vLLM 0.24.0, prefix caching off (scripts/modal_connector_serve.py, scripts/modal_connector_xinstance.py); Qwen3-0.6B 8k on A10G (scripts/modal_bench_resume.py). 2026-07-09.
- **Source:** /home/user/dexa/docs/RESULTS.md 'Update (2026-07-09, latest): the vLLM connector works end-to-end' and 'Cross-instance confirmed'; /home/user/dexa/docs/CONNECTOR_COMPLETION.md
- **Caveats:** TP=1, single attention backend, single-step prefill only (chunked prefill not saved), block-granular reuse; not validated across attention backends or GPU archs. README Phase-0 gate 'match or beat LMCache on TTFT and cross-replica reuse at 128K' is marked unmet.
- **Product relevance:** Supports 'portable, lossless session state across instances' at TP=1 on small models; not a speed claim.

## [PROVEN] connector-conformance — DexaConnector conforms to vLLM's V1 KVConnectorBase_V1 interface on a real release.

- **Headline:** 10/10 lifecycle-hook signatures match vLLM 0.24.0 with zero drift; tier-0 paged-block round-trip ok
- **Numbers:** tier 0: roundtrip true, store true | tier 1: subclass_real_base true, n_match 10/10 (get_num_new_matched_tokens, update_state_after_alloc, build_connector_meta, request_finished, register_kv_caches, start_load_kv, wait_for_layer_load, save_kv_layer, wait_for_save, get_finished)
- **Setup:** A10G, vLLM 0.24.0, benchmarks/vllm_connector_check.py via scripts/modal_scale_and_connector.py --only connector. 2026-07-09.
- **Source:** /home/user/dexa/benchmarks/results/2026-07-09-vllm-connector-check.json; /home/user/dexa/docs/RESULTS.md 'vLLM connector validated against a real vLLM (0.24.0)'
- **Caveats:** Signature match only; a 3-arg constructor requirement in vLLM >=0.24 was discovered later via a live probe (docs/CONNECTOR_COMPLETION.md). Behavior validated separately (cross-instance-resume).
- **Product relevance:** Interface compatibility with one pinned vLLM version; vLLM's connector API is noted as unstable.

## [OPEN] chunked-prefill-gap — The connector persists long (multi-step / chunked) prefills, so long sessions can be resumed.

- **Headline:** 16k run saved nothing (only an 8-token warmup) without the max_num_batched_tokens workaround -> resume 1.00x
- **Numbers:** 1.00x at 16k; save loop scans only scheduled_new_reqs, not scheduled_cached_reqs.
- **Setup:** scripts/modal_bench_resume.py, Qwen3-0.6B, A10G, vLLM 0.24.0. 2026-07-09.
- **Source:** /home/user/dexa/docs/RESULTS.md session-resume update ('critical gap surfaced'); /home/user/dexa/docs/CONNECTOR_COMPLETION.md 'Remaining (honest)'
- **Caveats:** Repo calls this a blocker because the estimated ~64k+ crossover regime cannot be single-step prefilled. TP>1 and cross-backend portability also unimplemented.
- **Product relevance:** Long-session persistence through the connector is not yet demonstrable; also blocks the 64k benchmark that would test the load-vs-prefill crossover.

## [BOUNDED] attention-matching-vs-h2o — Attention Matching (analytic KV compaction with bias + value refit) is a better general-purpose compactor than H2O / SnapKV.

- **Headline:** AM wins only at extreme compression (>=128x single-needle, >=512x multi-needle); at 128x multi-needle on 8B, H2O 0.82 vs AM 0.53, and H2O is ~5x cheaper to compact
- **Numbers:** SmolLM2-360M single-needle, 8 seeds, T~816: 128x AM 0.902 vs HH 0.426 vs SnapKV 0.615 (paired delta +0.476, p=0.001, 8/8 seeds); 4-32x pooled AM 0.977 vs HH 0.978 (p~0.94); compaction time AM 1.29 s vs HH 0.25 s. Llama-3.1-8B 8k single-needle (4 seeds, mass_frac=1.0, configs/extreme-8b.yaml): AM/H2O/SnapKV 128x 0.96/0.90/0.90; 512x 0.95/0.86/0.87; 1024x 0.96/0.81/0.78 (AM paired-win 4/4). Llama-3.1-8B multi-needle 8k+16k (6 seeds, configs/harden-8b.yaml): 128x 0.53/0.82/0.86 (AM win 25%); 256x 0.53/0.60/0.60 (67%); 512x 0.55/0.41/0.42 (92%); 1024x 0.51/0.33/0.32 (92%). Pure-importance AM collapsed to ~0.04 needle recall at 8000 tokens before the mass-aware hybrid fix. Toy FakeBackend recon cosine: AM 0.996 vs random 0.815 vs recent 0.778 (doc: sanity check, not evidence).
- **Setup:** benchmarks/niah_real.py (SmolLM2, CPU); 8B runs per configs/extreme-8b.yaml and configs/harden-8b.yaml; dated 2026-06-28 (8B) and earlier (SmolLM2), imported in the initial commit.
- **Source:** /home/user/dexa/docs/RESULTS.md 'Update (2026-06-28)' and sections 1-2; /home/user/dexa/docs/CLUSTER.md section 8
- **Caveats:** recall_frac is an affine log-prob rescaling that can exceed 1.0 (denoising artifact); self-study reference queries can peek at the answer; AM not compute-matched; 8 seeds cannot resolve mid-ratio gaps; independent sweeps show HH ahead at 16-64x. The single-needle vs multi-needle reversal at 128x is recorded in the same doc.
- **Product relevance:** Supports only 'AM degrades gracefully at >=512x'; the repo concludes the compaction algorithm is not the differentiator.

## [BOUNDED] long-horizon-agentic-working-memory — Bounded working memory (iterative compaction) retains late recall better than recent-window truncation at fixed memory over a multi-turn trajectory.

- **Headline:** SmolLM2-360M: late recall full_kv 1.000 / heavy_hitter 0.225 / attention_matching 0.156 / truncate_recent 0.131, at 465 vs 1252 peak tokens
- **Numbers:** full_kv 1.000, 1252 tok, 102.6 MB, 0 compactions, 0.0 s | truncate_recent 0.131, 300 tok, 24.6 MB | dexa:attention_matching 0.156, 465 tok, 38.1 MB, 7 compactions, 39.7 s | dexa:heavy_hitter 0.225, 465 tok, 38.1 MB, 7 compactions, 30.9 s
- **Setup:** benchmarks/agentic_real.py on HuggingFaceTB/SmolLM2-360M-Instruct (CPU); facts planted up to 8 turns before the query. The 8B config (configs/agentic-8b.yaml) was not run.
- **Source:** /home/user/dexa/docs/RESULTS.md section 3 and 'Still open (needs GPU)'
- **Caveats:** Absolute late recall is low for every bounded method (0.13-0.23); small model; H2O beats AM here; 8B run explicitly unrun.
- **Product relevance:** Supports only a relative claim (bounded memory > truncation) on a 360M model.

## [OPEN] lmcache-reuse-vs-compaction-framing — Reuse/tiering (LMCache-style) saves compute but does not bound memory; compaction bounds memory at fixed budget.

- **Headline:** Simulated: LMCache-style footprint grows 22 KB -> 196 KB at 0.74 prefix hit rate vs WorkingMemory flat at ~32 KB after 7 compactions
- **Numbers:** 22 KB -> 196 KB vs ~32 KB budget; 0.74 hit rate; exact reuse/recompute counts from FakeBackend; NVMe latency modeled.
- **Setup:** bench/lmcache_baseline.py on the torch-free FakeBackend with CostModel defaults; not a real LMCache run.
- **Source:** /home/user/dexa/docs/RESULTS.md section 4
- **Caveats:** Simulated harness; absolute bytes and GPU-seconds are model-relative and modeled.
- **Product relevance:** A framing argument, not evidence against LMCache.

## [OPEN] cartridges-quality — A trained compact KV (cartridge, Eyuboglu et al. 2025) matches full-context QA quality at ~50-100x less memory and beyond the context window.

- **Headline:** On a 360M model on CPU the trained cartridge does not beat no-context on held-out QA; the GPU / real-model run has not been done
- **Numbers:** No accuracy table recorded. Training loop: KL converges to ~0; 82 tests passing. Pre-registered thresholds (docs/BENCHMARK.md) at 16-50x: F1 >= full_context - 2, >= best training-free method + 5, >= RAG + 3.
- **Setup:** dexa.cartridge compiler (KL distillation through a frozen model, warm-start from downsampled corpus KV), 360M model on CPU with generic-prompt or corpus-LM self-study data.
- **Source:** /home/user/dexa/docs/CARTRIDGES.md 'Status (v0.1 - honest)'; /home/user/dexa/docs/BENCHMARK.md; /home/user/dexa/README.md status paragraph
- **Caveats:** Repo attributes the failure to the self-study data distribution (needs corpus-conditioned synthetic Q&A from a capable model + GPU). STILL training only smoke-tested on tiny-random Llama (KL decreases over ~12 steps). README: 'high-ratio compaction loses multi-fact fidelity'.
- **Product relevance:** No quality evidence for cartridges yet; the make-or-break experiment is unrun and the one small-scale attempt was negative.

## [BOUNDED] incremental-recompute — Segment-level incremental recompute reduces tokens reprocessed per mutating turn vs full re-prefill, with equivalent outputs.

- **Headline:** 4.6x fewer tokens reprocessed over a 20-step simulated agent loop (tiny-random Llama, CPU); equivalence unit-tested
- **Numbers:** 4.6x (README only; no results file). Tests assert reused prefix bit-identical (np.array_equal), recomputed K/V allclose atol 1e-5 rtol 1e-4 to a full prefill, and identical 8-token greedy continuation; append reuses all prior tokens, mid-context edit reuses only the system prefix.
- **Setup:** benchmarks/incremental_recompute_bench.py with hf-internal-testing/tiny-random-LlamaForCausalLM, CPU, 20 steps; tests/test_incremental_recompute.py.
- **Source:** /home/user/dexa/README.md Phase 1 'Gate (measured)'; /home/user/dexa/tests/test_incremental_recompute.py; /home/user/dexa/benchmarks/incremental_recompute_bench.py
- **Caveats:** Token-count metric only (hardware-independent); wall-time indicative; tiny random model; no recorded run output in the repo; an edit early in a long context still recomputes everything after it.
- **Product relevance:** Supports a token-reprocessing reduction on a simulated loop; no GPU wall-time evidence.

## [BOUNDED] selective-recompute-hkvd-rope — CacheBlend-style HKVD selection recovers downstream quality faster than recency/random when recomputing a fraction of stale tokens; RoPE re-phasing handles length-changing edits exactly.

- **Headline:** HKVD ordering beats recency and random at every recompute level on random weights (recency worst); RoPE re-phase reconstructs shifted keys to atol 1e-5; the '~15% recovers most' magnitude is untested
- **Numbers:** No numeric table recorded. Test tolerances: keys atol 1e-5 / rtol 1e-4, values atol 1e-6, zero-delta identity atol 1e-6.
- **Setup:** benchmarks/selective_recompute_bench.py on hf-internal-testing/tiny-random-LlamaForCausalLM (CPU), strategies hkvd/recent/random; tests/test_rope_rephase.py, tests/test_selective_engine.py.
- **Source:** /home/user/dexa/README.md Phase 1 'Selective recompute' and 'Exact RoPE re-phasing' bullets; /home/user/dexa/tests/test_rope_rephase.py
- **Caveats:** Random weights only; the compute realization (forward only selected tokens) is not built, so there is no wall-time saving yet.
- **Product relevance:** Supports mechanism correctness; no speed or quality-magnitude claim.

## [OPEN] persist-format-blob — The memory-mapped blob format loads persisted KV faster than the .npz container.

- **Headline:** '~1.7x faster resume load vs .npz' (README); no recorded run output in the repo
- **Numbers:** 1.7x (README sentence only).
- **Setup:** benchmarks/persist_format_bench.py on a synthetic KV slab (no model/GPU), numpy fallback on a laptop without the Rust toolchain.
- **Source:** /home/user/dexa/README.md 'Memory-mapped blob format'; /home/user/dexa/benchmarks/persist_format_bench.py
- **Caveats:** No results JSON; synthetic slab; the later bf16 keep_native fix changed the load path materially (docs/RESULTS.md).
- **Product relevance:** Minor engineering claim without a recorded artifact.
