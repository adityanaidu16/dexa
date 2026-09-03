# D. Document-VLM inference provider

*An OpenAI-compatible endpoint serving an open VLM (Qwen2.5-VL-7B measured) at a published accuracy-per-dollar operating point for document pages, sold on a same-pages rerunnable benchmark; the technical lever is a resize config flag, the dollar side is modeled, and the in-engine moat is blocked.*

**Evidence grade C.** The technical lever (visual-token budget vs accuracy frontier: 2.7x throughput at -1.0 pt from 1536px to 1024px) was measured once, in-engine (vLLM 0.24.0, A100-80GB), on 200 DocVQA pages, one run, relaxed match, offline eager batch — B-grade on its own. But the brief's core value claim is bounded and contradicted: the accuracy-parity half rests on a 200-page relaxed-match head-to-head against a 2024 model (GPT-4o) whose run output was never recorded beyond one number; the cost half has never been measured in dollars (the repo's headline '~40x cheaper' is a pricing model whose 56px-per-token formula contradicts the repo's own measured 1,017/2,201 prompt tokens, collapsing to ~9-10x at the same assumed rates); the moat (in-model pruning) is blocked with no result; and against a same-config vLLM baseline the engine advantage is 1.0x by construction.

**Weeks to first credible public proof:** 3-5 weeks to the first credible public proof (estimate): an ANLS benchmark on >= 1,000 DocVQA validation pages with bootstrap CIs, Qwen2.5-VL-7B and Qwen3-VL-8B on vLLM in serving mode with logged GPU-seconds, versus gpt-5.4 / gpt-5.6-luna / Gemini 3.8 Flash (medium) / Claude Haiku 4.5 / DeepSeek V4-flash-vision with usage-based $/1K pages, plus Textract Queries and Azure query fields as native OCR-QA baselines, published with the harness. Assumptions: 1-2 engineers; vendor accounts for Azure/GCP/AWS/DeepSeek obtained in week 1 (the main slip risk); Qwen3-VL-8B loads on vLLM 0.24.0 (unverified in-repo); roughly 15-25 A100-hours (estimate; at Modal's $2.50/hr about $40-65) plus a few hundred dollars of API spend (estimate). The moat spike is off this path. A sellable metered endpoint (auth, payments, batch queue, SSE, SLA) is 8-10 weeks (estimate), given the gaps in 03-build-inventory.md (no OAuth callback, no payments, Dockerfile omits control/, stream forced off).

---

## 1. Thesis

A document-VLM inference provider: an OpenAI-compatible endpoint serving an open vision-language model (Qwen2.5-VL-7B measured; Qwen3-VL-8B is a founder decision) at a published, measured accuracy-per-dollar operating point for document pages, sold on a same-pages, rerunnable benchmark against cloud OCR, VLM parsers and frontier models, with per-page billing as one pricing option (section 11). The buyer already pays per page (Azure DI, Google Document AI, Textract, Reducto, LlamaParse, Extend, Unstructured, Mistral OCR) or per image token (OpenAI, Anthropic, Gemini, DeepSeek). The sentence a customer would repeat: "On our own pages it scored at least as well as the frontier model, it is priced like our OCR bill, and we reran their benchmark ourselves."

What today's evidence supports is narrower. On 200 DocVQA validation pages with a relaxed scorer, a 7B open model at a 1024px budget scored 0.925 against GPT-4o's 0.880, and 1024px ran 2.7x more pages per GPU-second than 1536px for a 1.0-pt loss. The dollar side is modeled, not measured; the in-engine moat is blocked; and against a self-hosted vLLM with the same resolution flag the throughput advantage is 1.0x by construction (evals/RESULTS.md calls the lever a client-doable resize). Honest framing: an operating-point, benchmark and billing product first; an engine product only if the moat sprint in section 6 passes.

## 2. Customer and workload

**Who buys.** Intelligent-document-processing teams (accounts payable, lending, claims, legal review) and parsing startups that resell to them. No customer interviews exist in either repo; the workload shape below is inferred from vendors' pricing pages.

**Inferred shape.** One page image per request, stateless, batchable, backlog-concentrated; vendors price for it (Mistral OCR batch at half price, OpenAI and Anthropic Batch at 50% off, Reducto throughput tiers of 200/350/500+ concurrent pages; 05-research-docai.md). Unsupported by any research file: real page mix, resolutions, output type (short answers vs transcription vs extraction), concurrency, idle pattern. Contracts are multi-page; the measured config is one image per prompt at max-model-len 8192.

| Path | Engine / model | Tokens or unit per page | How they pay |
|---|---|---|---|
| Cloud OCR and extraction | Azure DI / Google Document AI / Textract | 1 page | OCR $1.50 per 1K ($0.60 above 1M Azure, 5M Google); extraction $30 / $30 / $50 per 1K (05-research-docai.md) |
| VLM parsers | Reducto r-1, Unstructured, Extend, LlamaParse | 1 page or credits | $0.01/page (r-1, 2026-09-01); $0.015/page; $0.025/page Parse PAYG; $1.25 per 1K credits at 1-45 credits/page |
| Frontier VLM | gpt-5.4; Gemini 3; Claude; DeepSeek V4-flash-vision | 1,229 tokens for 1024x1024 (gpt-5.4); 560/page at medium (Gemini); 1,296 for 1000x1000 plus 1,500-3,000 text tokens/page (Claude); up to 384/image (DeepSeek) | per input token (05-research-docai.md, 05-research-caching.md) |
| Self-hosted open VLM | Qwen3-VL on vanilla vLLM or SGLang (Qwen3-VL README) | measured on Qwen2.5-VL-7B: 2,201 prompt tokens at 1536px, 1,017 at 1024px, 574 at 768px (avg prompt tokens incl. question and template) | GPU-hours |

Outputs were 32 tokens in the QA eval; transcription would run 1,500-3,000 tokens per page (Claude's PDF docs, a proxy), where the visual-token lever does not reduce per-token decode work; unmeasured (section 10).

## 3. The pain, in the customer's words

Paraphrases of what pricing and benchmark facts imply; not quotes.

- "We pay $1.50 per thousand pages for OCR and $30 per thousand again for extraction, and it still misses fields." (Azure, Google list prices.)
- "The frontier model reads our pages better, but nobody can tell me what a page costs until the bill arrives." (OpenAI: 32px patches times a multiplier; Claude: 28px patches plus PDF text; Gemini: 258 tokens per 768px tile or fixed levels; DeepSeek: up to 384 per image.)
- "We self-host Qwen on vLLM, but we do not know whether our resolution setting costs us accuracy or 3x throughput." (Measured: 1536px 0.935 vs 1024px 0.925 at 2.7x vs 768px 0.885 at 4.1x.)
- "Every parser publishes its own benchmark on its own pages." (olmOCR-Bench, OmniDocBench and the Nanonets leaderboard name different leaders; Together lists Qwen3-VL-32B at DocVQA 93.3, the Qwen3-VL report 96.9.)

## 4. Value proposition and the proof-of-value benchmark

**Value proposition.** Frontier-class document accuracy from an open model at a published per-page operating point, with the harness open so the buyer reruns it on their own pages. Not, today, a faster engine: the measured lever is `--mm-processor-kwargs max_pixels` or a resize before upload.

**The seed (measured).** Same 200 DocVQA validation pages, same relaxed scorer: Qwen2.5-VL-7B at 1024px scored 0.925 (1,017 prompt tokens/page, 30.2 img/s offline, one A100-80GB) versus GPT-4o at 0.880 (detail high, 2048px cap, temperature 0, max_tokens 32) (01-evidence-ledger-dexa.md, doc-vlm-frontier and gpt4o-docvqa-head-to-head; commits 7237401 and c729976, 2026-08-04). GPT-4o's token usage and $/1K pages were never recorded.

**The benchmark a skeptical buyer would believe.** Metric: accuracy per dollar per page, as (a) ANLS on DocVQA validation (the vendor-table metric; the repo's relaxed match, "normalized exact match or gold substring in prediction", is more lenient), (b) olmOCR-Bench or OmniDocBench for transcription, (c) $/1K pages measured from GPU-seconds consumed in serving mode at a stated utilization and from incumbents' returned usage counters, never list-price arithmetic. Setup: N >= 1,000 pages, one harness, same pages, same day, prompts and scorers published, one `modal run`, bootstrap confidence intervals (at N=200 one point is two pages).

Baselines in two classes. Class 1, same weights, decides whether there is an engine claim: vanilla vLLM and SGLang serving the same Qwen weights at the same max_pixels on the same GPU. The expected result is 1.0x; an engine claim must beat this, not vLLM's default resolution. Class 2 decides the product claim: Qwen3-VL-8B on vLLM; GPT-4o (0.880 measured); gpt-5.4, gpt-5.6-luna, Gemini 3.8 Flash at medium, Claude Haiku 4.5, DeepSeek V4-flash-vision; the native OCR question-answering endpoints, Textract Queries ($15/1K) and Azure query fields ($10/1K); Google Document AI OCR plus a text LLM (Document AI lists no QA endpoint); Reducto r-1, Mistral OCR 4.1, olmOCR-2-7B and dots.mocr for transcription (the last two vendor-reported at 82.4 and 83.9 olmOCR-Bench). Every Class 2 system is unmeasured here.

Pre-registered targets: ANLS within 1 pt of the 1536px point at >= 2.5x pages per GPU-hour in serving mode (reproducing the offline 2.7x); ANLS at or above every frontier cheap tier on the same pages, or within 1 pt at <= 1/3 of its usage-measured $/page; measured $/1K published with utilization stated. Why believable: same pages, open weights, open harness, the buyer's own PDFs.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| OpenAI-compatible gateway | Custom (dexa_platform/gateway/app.py, 261 lines, FastAPI; built for screenshots) | Keys, forwarding, per-request cost telemetry, usage ledger; forces stream=False at app.py:127 |
| Per-page cost meter | Custom (dexa_platform/gateway/pricing.py; Qwen formula wrong, section 6) | Image dimensions -> billed pages and comparison costs |
| Page preprocessor and budget policy | New | PDF rasterization; enforce a 1024 long-side resize (the measured point); adaptive budget (unproven) |
| Serving engine | OSS: vLLM 0.24.0 (serve/modal_doc_vlm_serve.py: `vllm serve`, max_pixels 1,048,576, one image per prompt, max-model-len 8192, `max_inputs=32`, scale-to-zero) | Prefill/decode; unmodified |
| Model | OSS weights: Qwen2.5-VL-7B-Instruct (measured); Qwen3-VL-8B (Apache-2.0, vendor-reported; vLLM 0.24.0 support unverified) | The accuracy |
| Batch queue | New | Async page jobs |
| Benchmark harness | Custom (evals/modal_vlm_frontier.py, evals/modal_incumbent_docvqa.py) | The proof of value; needs ANLS, incumbent adapters, serving-mode timing |
| Control plane | Custom (dexa_platform/control: keys, daily rollups, credit ledger; no OAuth callback, no payments) | Signup, keys, metering |
| In-model visual-token pruning | Custom fork of the HF forward pass (evals/modal_vlm_moat.py, transformers 4.49.0) | The intended moat; blocked, no result |

**Where differentiation lives today:** configuration, billing and the benchmark, not a kernel, scheduler or connector; the 2.7x is a naive resize the repo says "sizes the prize but isn't yet the moat" (evals/RESULTS.md). The aspirational differentiation is inside the forward pass: vLLM ships only `--video-pruning-rate` (EVS, VidCom2); RFC #45098 for `--image-pruning-rate` is open with Qwen3-VL as its example; SGLang's EVS "cannot work with VLMs that use positional embeddings [Such as Qwen2.5VL]" (05-research-docai.md). If the RFC merges, image pruning becomes an OSS flag. Built on vLLM: everything. Replaced: nothing.

```
 PDF/images --> [Rasterize + 1024px budget policy] --> [Gateway: keys, per-page meter, batch queue]
                     (New)                              (dexa_platform/gateway + control)
                                                                    |
                                                                    v
                                  [vLLM 0.24 OpenAI server, Qwen2.5-VL-7B / Qwen3-VL-8B]
                                  (serve/modal_doc_vlm_serve.py; unmodified engine)
                                                                    |
                              (moat sprint, unproven) [pruned-grid mRoPE forward pass]
                                                                    |
 Benchmark harness (evals/*) <---- same pages ----> incumbents (OCR APIs, parsers, frontier VLMs)
```

## 6. Evidence

### Proven

- Visual-token budget vs accuracy frontier on documents: 1536px 0.935 / 11.4 img/s / 2,201 tokens; 1024px 0.925 / 30.2 / 1,017 (2.7x, -1.0 pt); 768px 0.885 / 47.2 (4.1x, -5.0 pts); 512px 0.735 (6.9x); 384px 0.445 (8.6x). Qwen2.5-VL-7B, vLLM 0.24.0, A100-80GB, 200 DocVQA validation pages, relaxed match, one run, temperature 0, warm-up batch, `enforce_eager=True`, one offline `llm.chat` batch of 200 per budget, max_tokens 32 (01-evidence-ledger-dexa.md, doc-vlm-frontier). Caveats: the "visual tokens" column is average prompt tokens including question and template; -1.0 pt is 2 pages; throughput is offline batch, not serving.
- A deployable endpoint config exists: `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --mm-processor-kwargs {"max_pixels":1048576}` on Modal A100-80GB (serve/modal_doc_vlm_serve.py, commit 85d292b); a BYOC compose file runs the same model on vllm-openai v0.24.0. Caveat: max_pixels 1,048,576 is not the measured 1024px point; a full-resolution page of DocVQA-like aspect smart-resizes to about 896x1148 = 1,312 tokens (arithmetic from pricing.py at 28 px/token), an unmeasured point between the 1024 and 1536 rows, unless the client pre-resizes to <= 1024 long side. Never exercised under load.
- dexa_platform tests: 29 of 29 pass (accounts 7, pricing 5, redundancy 5, sessions 7, tenants 5). The gateway is a screenshot gateway; test_pricing.py pins the wrong 56px formula (`big <= 325`).

### Bounded / contradicted

- **GPT-4o head-to-head, 0.925 vs 0.880.** Same pages, same scorer; but relaxed match, 200 pages (the gap is 9 pages), a 2024 model, and the run's token usage, $/1K pages and GPT-4o-mini accuracy were never recorded (gpt4o-docvqa-head-to-head). ANLS unmeasured.
- **"~40x cheaper than GPT-4o" is a model that contradicts the repo's own token count.** pricing.py computes 322 Qwen tokens for a 1280x800 screenshot with a 56-px-per-token formula (patch 28, merge 2) and bills an assumed $0.20/M: (1,105/322) x (2.50/0.20) = 42.9x. The frontier run measured 1,017 prompt tokens at 1024px and 2,201 at 1536px, consistent with 28 px/token (a 791x1024 page = 1,036), not 56 (252). At 28 px/token the same functions give 10.4x for the screenshot and 9.2x for the page, all of it the assumed price ratio (04-conflicts.md conflict 10; reproduced this session). Serving cost has never been measured in dollars.
- **"$0.02-0.06 / 1k pages" is a hard-coded string** in evals/modal_incumbent_docvqa.py. Derived: 30.2 pages/s = 108,720 pages/hr; at the repo's assumed $1.80/hr (evals/stateful_cost_model.py:20) that is $0.017/1K at 100% utilization and 32-token outputs; at Modal's actual A100-80GB rate, $2.50/hr, $0.023; at RunPod's $1.39-1.59, $0.013-0.015 (05-research-gpu-pricing.md); the 1536px point is $0.034-0.061. Utilization, serving mode, cold starts (scale-to-zero, 900 s startup timeout) and transcription outputs are unmeasured.
- **Qwen3-VL numbers are vendor-reported and cross-table.** Qwen3-VL-8B-Instruct DocVQA 96.1 is Table 4 of arXiv 2511.21631; GPT-5 (high) 91.5 is Table 2; the 8B table's comparator is GPT-5 nano at 88.2. Together lists Qwen3-VL-32B at 93.3 vs the report's 96.9. None reproduced here.
- **No measurement compares against vanilla vLLM.** The 2.7x is between two non-default resolutions of one engine; at the same max_pixels the difference is nil by construction. Throughput is offline batch; p50/p95 under concurrency unmeasured.

### Unproven

| Claim | Experiment that proves or kills it | Est. cost (estimates) |
|---|---|---|
| In-model pruning beats resize at matched token budget (the moat) with competitive throughput | Reimplement `get_rope_index` for a pruned set in the HF path (blocker: [3,1671] positions vs [3,881] masked); run evals/modal_vlm_moat.py at FRACTIONS 0.5/0.25, n >= 200 (default 80); add img/s per method (the prune path still runs the ViT at 1280px). Proceed if best prune minus resize >= +0.02 (script verdict) and prune img/s >= 0.8x resize | 6-15 A100-hours; 5-10 days, plus a vLLM port |
| Qwen3-VL-8B loads in vLLM 0.24.0, reproduces >= 93 ANLS, holds the frontier | Rerun modal_vlm_frontier.py with Qwen3-VL-8B, ANLS, N >= 1,000 | 4-8 A100-hours; 2 days |
| The 0.925 vs 0.880 gap survives ANLS and N >= 1,000 | Re-score archived predictions; rerun both on 1,000 pages | 2 A100-hours plus API spend; 2 days |
| The frontier holds for transcription-length outputs | Transcription prompt, max_tokens 1,500; pages/GPU-hour per budget | 4 A100-hours; 2 days |
| Serving-mode $/1K pages at realistic utilization | `vllm bench serve`-style load at 8/32/128 concurrency; GPU-seconds/page, p50/p95, cold-start share | 6 A100-hours; 3 days |
| The endpoint's 1.05-megapixel point (about 1,312 tokens) matches the 1024 row | Add 1148px to BUDGETS | 1 A100-hour |
| Adaptive per-page budgeting beats fixed 1024px | Route by pixel density / text size to 768/1024/1536 | 4 A100-hours; 3 days |
| Frontier cheap tiers and OCR-QA endpoints on the same pages | Extend modal_incumbent_docvqa.py; usage-based $/1K | API spend; 3-5 days incl. onboarding |
| Buyers move for a per-page price and a benchmark | 5 design partners rerun the harness on their pages; count conversions | 4 weeks |

## 7. MVP and 6-week build plan

**What ships first:** `/v1/chat/completions` (model `doc-vlm`), a batch endpoint for PDFs, per-page metering with API keys, and a public benchmark page with the harness. Public proof by week 3, metered endpoint by week 6; payments, OAuth and an SLA are outside it (03-build-inventory.md gaps).

- **Week 1 - benchmark integrity.** Add ANLS and bootstrap CIs to evals/modal_vlm_frontier.py; add 1148px to BUDGETS; run Qwen2.5-VL-7B and Qwen3-VL-8B on >= 1,000 pages in serving mode (CUDA graphs on), GPU-seconds logged; add the transcription-output run. Reuse: modal_vlm_frontier.py, the `dexa-hf-cache` volume. New: ANLS scorer, GPU-hour logger. Slip risk: Qwen3-VL on vLLM 0.24.0 untested.
- **Week 2 - incumbents on the same pages.** Extend evals/modal_incumbent_docvqa.py to gpt-5.4, gpt-5.6-luna, Gemini 3.8 Flash (medium), Claude Haiku 4.5, DeepSeek V4-flash-vision, Textract Queries, Azure query fields, and a Document AI OCR + LLM pipeline; record returned usage. Nine adapters; vendor accounts from day 1. Reducto r-1 and Mistral OCR as transcription baselines. Slip risk: cloud OCR onboarding.
- **Week 3 - publish v0.** Benchmark page plus harness; measured $/1K from Week-1 GPU-seconds at stated utilization. The first credible public proof.
- **Week 4 - endpoint hardening.** Reuse serve/modal_doc_vlm_serve.py, dexa_platform/gateway/{app,tenants,store}.py, dexa_platform/control. Fix pricing.py to patch 14 / merge 2 and rewrite test_pricing.py; add a per-page meter; remove `stream=False` at app.py:127 and add SSE; add control/ to the Dockerfile; enforce the 1024 long-side resize.
- **Week 5 - PDF and batch.** New: rasterizer, budget policy, batch queue with an off-peak price (Morph's Standby bills 50% when fleet capacity is under ~25%; DeepSeek off-peak is half price). Reuse dexa_platform/dashboard/index.html. Design-partner onboarding; replace the hard-coded URL in serve/demo_client.py.
- **Week 6 - time-boxed moat spike, if the founder chooses it (section 11).** Pruned-grid mRoPE in the HF path; run the gate; write the result either way.

## 8. Pricing model

Options, not a decision: per page by output type (QA/extraction vs transcription) with a batch tier; per token; or per-GPU-hour dedicated. Per page is the unit every incumbent bill uses and is expressible here because a fixed budget bounds tokens per page (1,017 at 1024px), so pages per GPU-hour is a measurable constant (30.2 pages/s offline, eager, 32-token outputs). Per page keeps the resolution gain on the provider's side; a per-token provider's revenue falls when it cuts tokens. Gemini already prices a PDF page at a fixed 560 tokens at medium, so fixed per-page cost is not unique.

Modeled compute floor (unmeasured in production): $0.013-0.023 per 1K pages at 100% utilization for QA-length outputs at $1.39-2.50/GPU-hour (RunPod to Modal A100-80GB; 05-research-gpu-pricing.md); transcription unmeasured. Anchors: Azure Read $1.50/1K; Reducto r-1 $10/1K; Unstructured $15/1K; Gemini 3.8 Flash medium page about $0.42/1K input-only; gpt-5.6-luna about $0.25/1K for 1024x1024; DeepSeek V4-flash-vision about $0.17/1K images at peak, $0.08 off-peak (384 tokens at $0.44 and $0.22/M; derived, input only); Qwen3-VL-8B hosted at $0.20/M (secondary aggregator) times 1,017 tokens is about $0.20/1K pages, though Qwen3-VL's own per-page token count is unmeasured. Where a list price sits, and whether to publish the floor, are founder decisions.

## 9. Competitive facts

| Who | What they ship that is adjacent | What they do not ship (per the research files) | Source |
|---|---|---|---|
| Azure DI / Google Document AI / AWS Textract | Per-page OCR ($1.50/1K), layout ($10/1K), extraction ($30/1K), forms ($50/1K Textract); native QA: Textract Queries $15/1K, Azure query fields $10/1K | An open-weight model; a same-pages benchmark against VLMs; a Document AI QA endpoint | 05-research-docai.md |
| Reducto | r-1 at $0.01/page (2026-09-01), claims 20% lower error than its legacy pipelines; throughput tiers 200/350/500+ | Model size, latency, self-hosting | https://reducto.ai/blog/parse-r-1-model |
| LlamaParse / Extend / Unstructured | $1.25 per 1K credits; $0.025/page; $0.015/page | Disclosed underlying VLM | 05-research-docai.md |
| Mistral | OCR 4.1 $4/1K ($2 batch), self-reported olmOCR-Bench 85.20; self-hosted for enterprise | Open weights for OCR 4 | https://mistral.ai/news/ocr-4/ |
| OpenAI / Anthropic / Google | Frontier VLMs with published image-token rules; Gemini fixed 560 tokens per PDF page at medium, native PDF text uncharged | Open weights; per-page pricing (OpenAI, Anthropic) | 05-research-docai.md |
| DeepSeek | V4-flash-vision-exp (2026-08-21), images up to 384 tokens at V4-Flash pricing, off-peak half price, disk cache | Document benchmark numbers in the research files | https://api-docs.deepseek.com/news/news260821 |
| Together / DeepInfra / Fireworks | Host Qwen3-VL (32B $0.50/$1.50; 235B $0.20/$0.88; 8B $0.20/M per aggregator) | Per-page pricing; document-tuned operating points; same-pages benchmark | 05-research-docai.md |
| Qwen3-VL / InternVL3.5 / DeepSeek-OCR / olmOCR-2 / dots.mocr | Open weights; InternVL3.5's ViR halves visual tokens in-model; DeepSeek-OCR 97% precision under 10x compression; olmOCR-2-7B is a Qwen2.5-VL-7B fine-tune | A hosted per-page service (no primary listing found for the last four) | 05-research-docai.md |
| vLLM / SGLang | Video-token pruning (EVS, VidCom2); image-pruning RFC #45098 open (Qwen3-VL example) | Image-token pruning; EVS incompatible with Qwen2.5-VL positional embeddings | 05-research-docai.md |
| Morph, Wafer (analogs) | Specialized small models on a custom stack, 50% standby tier (Morph); engine/kernel tuning of open models on NVIDIA and AMD, ZDR (Wafer) | Document AI products | 05-research-morph.md; 05-research-wafer.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| Relaxed match flattered the GPT-4o gap | ANLS on the 200 pages, then 1,000 | Qwen minus GPT-4o < +1.0 pt | >= +2.0 pts, or parity with gpt-5.4-class at <= 1/5 measured $/page |
| Vendor-reported Qwen3-VL-8B does not reproduce | DocVQA ANLS on this harness | < 93.0 | >= 95.0 |
| No engine advantage exists | Pages/GPU-hour vs vanilla vLLM, same weights, same max_pixels, same GPU | 1.0x (expected) kills the "engine" pitch, not the product; the moat row decides the engine claim | > 1.3x |
| The moat sprint fails | Best prune minus resize at 25% tokens; prune img/s vs resize | <= 0.00, or img/s < 0.8x resize | >= +0.02 and img/s >= 0.8x |
| Serving cost is not what the model says | Measured $/1K pages at 50% utilization, serving mode, QA outputs | > $0.42/1K (Gemini medium-page anchor) | <= $0.05/1K (below the DeepSeek off-peak anchor) |
| The frontier collapses for transcription | Pages/GPU-hour 1024 vs 1536 with 1,500-token outputs | < 1.3x | >= 2.0x |
| Frontier cheap tiers price-match | luna, Gemini 3.8 Flash, DeepSeek V4-flash-vision ANLS and usage-based $/page | Any within 1 pt at <= 2x our measured price | All >= 3 pts below or >= 5x our price |
| vLLM RFC #45098 merges | Image pruning becomes an engine flag | Reframe the moat as a config again | RFC stalls |
| Buyers do not move for a benchmark | Design-partner conversions after rerunning on their pages | 0 of 5 | >= 2 of 5 |

## 11. Founder decisions

- **Market size and whether the per-page market is worth entering.** Cloud tier breakpoints (Azure $0.60/1K above 1M pages; Google above 5M) show volume exists but give no volumes; design-partner conversions would.
- **Whether Reducto r-1 at $0.01/page, Mistral OCR at $2-4/1K, DeepSeek V4-flash-vision at about $0.08-0.17/1K images, and cloud OCR tiers are adjacencies that matter.** None publishes a same-pages result against open VLMs.
- **Whether the proof must include an engine advantage.** Sell operating point + benchmark + billing with a 1.0x engine result stated openly, or require the moat gate first. Evidence: the moat run and the RFC's fate.
- **Hosted vs BYOC vs harness-only open source.** The repo has a Modal endpoint and a BYOC compose file; the harness could ship alone. Evidence: whether design partners want the endpoint or the config.
- **Model choice.** Qwen2.5-VL-7B (measured) vs Qwen3-VL-8B (vendor 96.1, unreproduced) vs a document fine-tune (olmOCR-2-7B, a Qwen2.5-VL-7B fine-tune at 82.4 olmOCR-Bench; training data about $0.12/page via Claude Sonnet 4) vs a transcription model (dots.mocr, DeepSeek-OCR).
- **The proof metric and accuracy bar.** DocVQA ANLS is saturated (vendor tables put open 8B above GPT-5); alternatives are olmOCR-Bench/OmniDocBench or a design partner's extraction F1. Parity with GPT-4o (2024) vs current cheap tiers.
- **GPU class and cloud.** All measurements are A100-80GB on Modal ($2.50/hr); RunPod A100 $1.39-1.59, H100 $2.69-3.95, MI300X $2.59-2.99, MI355X $2.29-2.95 (05-research-gpu-pricing.md, 05-research-wafer.md); the floor moves about 2x; unmeasured.
- **Moat sprint before, after, or never.** The repo recorded "gate the moat sprint on market validation"; the plan makes week 6 optional.
- **Product shape: QA/extraction vs transcription vs both.** Only QA is measured.
- **Per-page vs per-token vs dedicated pricing, the batch tier, and whether to publish the compute floor.**
- **Concurrency target and SLA.** Reducto sells 200/350/500+ concurrent pages; the endpoint is `max_inputs=32` per container with scale-to-zero; nothing measured.
- **Data policy.** Zero-data-retention (Morph and Wafer offer it) vs retaining pages for a benchmark corpus.

## 12. Combinations

- **Computer-use / screenshot VLM.** The proven screen-redundancy result (about 86% of 28px patches unchanged per action; realized reuse bounded at about 2-2.3x) uses the same encoder and pricing.py; one endpoint could serve pages and screenshots (cua-screen-redundancy, delta-tolerance-sweep).
- **Session-stateful serving.** Multi-question workflows over one document reuse a page prefix; vLLM in-GPU prefix-cache hits measured 12x at 4k and 25-34x at 16k TTFT on a text model, labeled table stakes by the ledger (vllm-warmstart-prefix-cache; 04-conflicts.md conflict 3); page-image prefix hits unmeasured.
- **Morph-style specialization.** Document fine-tune plus task-specific speculator is the playbook Morph and Relace describe for code (05-research-morph.md); olmOCR-2 shows the fine-tune half exists for this base.
- **Wafer-style hardware optimization.** Tuned engine configs on AMD parts could lower the floor; unmeasured for VLM prefill.
- **Benchmark-as-product.** The harness run publicly across vendors on the same pages is a standalone asset whichever direction wins.
- **Batch capacity sharing with voice or agent directions.** Page backlogs fill idle capacity; Morph's 50% standby tier is the precedent.
