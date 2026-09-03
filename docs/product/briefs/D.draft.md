## 1. Thesis

A document-VLM inference provider: an OpenAI-compatible endpoint that serves an open vision-language model (Qwen2.5-VL-7B today; Qwen3-VL-8B is a founder decision) at a measured accuracy-per-dollar operating point for document pages, billed per page, and sold on a same-pages, rerunnable benchmark against cloud OCR, VLM parsers, and frontier models. The buyer already pays per page (Azure Document Intelligence, Google Document AI, Textract, Reducto, LlamaParse, Extend, Unstructured) or per image token (OpenAI, Anthropic, Gemini) to read invoices, forms, and contracts. The sentence a customer would repeat: "On our own pages it scored at least as well as the frontier model, it is priced per page like our OCR bill, and we reran their benchmark ourselves." Today's evidence supports the first half of that sentence on 200 pages with a relaxed scorer; the second half is a plan, not a result.

## 2. Customer and workload

**Who buys.** Intelligent-document-processing teams (accounts payable, lending, insurance claims, legal review) and the parsing startups that resell to them. The workload is one page image per request, stateless, batchable, and concentrated in backlogs; vendors already price for this shape (Mistral OCR batch at half price, OpenAI and Anthropic Batch at 50% off, Reducto throughput tiers of 200/350/500+ concurrent pages; 05-research-docai.md).

**What they run today, concretely.**

| Path | Engine / model | Tokens or unit per page | How they pay |
|---|---|---|---|
| Cloud OCR | Azure DI Read / Google Enterprise OCR / Textract DetectDocumentText | 1 page | $1.50 per 1K pages; $0.60 above 1M (Azure) or 5M (Google) (05-research-docai.md) |
| Cloud extraction | Azure Custom / Google Form Parser / Textract Forms | 1 page | $30 / $30 / $50 per 1K pages (05-research-docai.md) |
| VLM parsers | Reducto r-1, Unstructured, Extend, LlamaParse | 1 page or credits | $0.01/page (r-1), $0.015/page, $0.025/page Parse PAYG, $1.25 per 1K credits at 1-45 credits/page (05-research-docai.md) |
| Frontier VLM | gpt-5.4 patch tokens; Gemini 3 PDF page; Claude 28px patches + PDF text | 1,229 tokens for 1024x1024 (gpt-5.4); 560 tokens/page at medium (Gemini); 1,296 tokens for 1000x1000 plus 1,500-3,000 text tokens/page (Claude) | per input token (05-research-docai.md) |
| Self-hosted open VLM | Qwen3-VL on vanilla vLLM or SGLang (per the Qwen3-VL README) | measured here: 2,201 prompt tokens at 1536px, 1,017 at 1024px, 574 at 768px on Qwen2.5-VL-7B | GPU-hours (evals/RESULTS.md 'Multimodal execution thesis') |

Context is short (one page plus a question is 1-2.3K tokens at the budgets above); outputs are 32 tokens in the QA eval and would be hundreds to thousands of tokens for transcription (Claude's PDF docs put a page's text at 1,500-3,000 tokens, a proxy; transcription serving cost is unmeasured). Concurrency, idle pattern, and real customer page mixes are unmeasured; the only corpus measured is 200 lmms-lab/DocVQA validation pages.

## 3. The pain, in the customer's words

No customer interviews exist in either repo; these are paraphrases of what the pricing and benchmark facts imply, not quotes.

- "We pay $1.50 per thousand pages for OCR and then $30 per thousand again for extraction, and the extraction still misses fields." (Azure and Google list $1.50/1K for OCR and $30/1K for custom extraction; 05-research-docai.md.)
- "The frontier model reads our pages better than OCR, but nobody can tell me what a page costs until the bill arrives." (OpenAI counts 32px patches times a per-model multiplier, Claude counts 28px patches plus PDF text tokens, Gemini bills 258 tokens per 768px tile or fixed media_resolution levels; 05-research-docai.md.)
- "We self-host Qwen on vLLM; it works, but we do not know whether our resolution setting costs us accuracy or 3x throughput." (Measured: 1536px 0.935 vs 1024px 0.925 at 2.7x vs 768px 0.885 at 4.1x; evals/RESULTS.md.)
- "Every parser publishes its own benchmark on its own pages." (olmOCR-Bench, OmniDocBench, and the Nanonets leaderboard report different leaders; Together lists Qwen3-VL-32B at DocVQA 93.3 while the Qwen3-VL report says 96.9; 05-research-docai.md.)

## 4. Value proposition and the proof-of-value benchmark

**Value proposition.** Frontier-class document accuracy from an open model at a published per-page operating point, with the benchmark harness open so the buyer can rerun it on their own pages.

**What is measured now (the seed).** On the same 200 DocVQA validation pages, same relaxed-match scorer: Qwen2.5-VL-7B at a 1024px long side scored 0.925 (1,017 prompt tokens/page, 30.2 img/s offline on one A100-80GB) versus GPT-4o at 0.880 (detail high, images capped at 2048px, temperature 0) (01-evidence-ledger-dexa.md, gpt4o-docvqa-head-to-head; evals/modal_incumbent_docvqa.py). The full sweep is in section 6 (evals/RESULTS.md 'Multimodal execution thesis'; commit 7237401, 2026-08-04).

**The benchmark a skeptical buyer would believe.** The exact metric is accuracy per dollar per page, reported as (a) ANLS on DocVQA (the metric every vendor table uses; the repo's relaxed match is "normalized exact match or gold substring in prediction", which is more lenient), (b) olmOCR-Bench or OmniDocBench for full-page transcription, and (c) measured dollars per 1K pages computed from GPU-hours actually consumed and from the incumbents' returned usage counters, never from list-price arithmetic. Setup: N >= 1,000 pages, one harness, every system on the same pages the same day, prompts and scorers published, reproducible from a single `modal run`. Baselines, named: vanilla vLLM serving the same Qwen weights at default resolution (the baseline that decides whether there is a product beyond a config flag); SGLang, same weights; Qwen3-VL-8B on vLLM; GPT-4o (0.880 measured here); gpt-5.4, gpt-5.6-luna, Gemini 3.8 Flash at medium, Claude Haiku 4.5, Azure DI Read+Layout, Google Document AI, Textract Queries, Reducto r-1, and Mistral OCR 4.1 (all unmeasured); olmOCR-2-7B and dots.mocr as open transcription baselines (vendor-reported 82.4 and 83.9 on olmOCR-Bench; 05-research-docai.md). Target numbers to pre-register: ANLS within 1 pt of the 1536px point at >= 2.5x pages per GPU-hour versus vanilla-vLLM default; ANLS >= GPT-4o on the same pages; measured cost per 1K pages published with the utilization assumption stated. Why believable: same pages, open weights, open harness, and the buyer can swap in their own PDFs.

## 5. Architecture

| Component | Custom or OSS | Role |
|---|---|---|
| OpenAI-compatible gateway | Custom (dexa_platform/gateway/app.py, 261 lines, FastAPI) | Key resolution, forwarding, per-request cost telemetry, usage ledger |
| Per-page cost meter | Custom (dexa_platform/gateway/pricing.py; needs the Qwen token formula fixed, see section 6) | Turns image dimensions into billed pages and comparison costs |
| Page preprocessor and budget policy | New | PDF rasterization; resolution/max_pixels policy that encodes the measured frontier; adaptive per-page budget (unproven) |
| Serving engine | OSS: vLLM 0.24.0 (serve/modal_doc_vlm_serve.py: `vllm serve`, max_pixels 1,048,576, one image per prompt, max-model-len 8192) | Prefill/decode; unmodified today |
| Model | OSS weights: Qwen2.5-VL-7B-Instruct (measured); Qwen3-VL-8B (Apache-2.0, vendor-reported) | The accuracy |
| Batch queue | New | Async page jobs |
| Benchmark harness | Custom (evals/modal_vlm_frontier.py, evals/modal_incumbent_docvqa.py) | The proof of value; extended with ANLS and incumbent adapters |
| Control plane | Custom (dexa_platform/control: keys, daily usage rollups, credit ledger; 29 tests pass) | Signup, keys, metering |
| In-model visual-token pruning | Custom fork of the model forward pass (evals/modal_vlm_moat.py attempt) | The intended moat; blocked, no result |

**Where differentiation lives today:** at the gateway and configuration layer (operating point, per-page billing, benchmark), not in a kernel, scheduler, or connector; the measured 2.7x is naive resize before the engine, which the repo itself says "sizes the prize but isn't yet the moat" (evals/RESULTS.md). The aspirational differentiation is inside the forward pass: vLLM ships only `--video-pruning-rate` (EVS/VidCom2) with RFC #45098 for `--image-pruning-rate` open and uncommented, and SGLang's EVS "cannot work with VLMs that use positional embeddings [Such as Qwen2.5VL]" (05-research-docai.md). Built on vLLM: everything. Replaced: nothing.

```
 PDF/images --> [Rasterize + budget policy] --> [Gateway: keys, per-page meter, batch queue]
                     (New; encodes the             (dexa_platform/gateway + control)
                      measured px->tokens frontier)          |
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

- Visual-token budget vs accuracy frontier on documents: 1536px 0.935 / 11.4 img/s / 2,201 tokens; 1024px 0.925 / 30.2 img/s / 1,017 tokens (2.7x, -1.0 pt); 768px 0.885 / 47.2 (4.1x, -5.0 pts); 512px 0.735 (6.9x); 384px 0.445 (8.6x). Qwen2.5-VL-7B, vLLM 0.24.0, A100-80GB, 200 DocVQA validation pages, relaxed match, warm-up batch, `enforce_eager=True`, offline `llm.chat` batches (01-evidence-ledger-dexa.md, doc-vlm-frontier; evals/modal_vlm_frontier.py).
- A deployable endpoint config exists at the measured operating point: `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --mm-processor-kwargs {"max_pixels":1048576}` on Modal A100-80GB (serve/modal_doc_vlm_serve.py, commit 85d292b, 2026-08-04); a BYOC compose file runs the same model on vllm-openai v0.24.0 (03-build-inventory.md, dexa_platform/docker-compose.byoc.yml).
- Gateway, pricing, tenant, and control-plane code runs: 29 of 29 dexa_platform tests pass (03-build-inventory.md, Tests run, item 4).

### Bounded / contradicted

- **GPT-4o head-to-head, 0.925 vs 0.880.** Same 200 pages, same scorer; but relaxed match, 200 pages, and the GPT-4o run's token usage, measured $/1K pages, and GPT-4o-mini accuracy were never recorded (01-evidence-ledger-dexa.md, gpt4o-docvqa-head-to-head). ANLS on the same pages is unmeasured.
- **"~40x cheaper than GPT-4o" is a model, and its token count contradicts the repo's own measurement.** pricing.py computes 322 Qwen tokens for a 1280x800 screenshot with a 56-pixel-per-token formula (patch 28, merge 2) and bills at an assumed $0.20/M blended rate, giving (1,105/322) x (2.50/0.20) = 42.9x (39.1x with 50 text and 10 output tokens). The frontier run measured 1,017 prompt tokens at 1024px and 2,201 at 1536px, which matches Qwen2.5-VL's 28-pixel-per-token geometry (a 791x1024 page = 1,036 tokens; a 1,186x1,536 page = 2,310), not 56px (252 and 320). Re-running pricing.py's own functions at 28px/token: the 1280x800 screenshot is 1,334 Qwen tokens vs 1,105 GPT-4o tokens, so the modeled ratio at the same assumed rates is 10.4x, and a 791x1024 document page is 1,036 Qwen tokens vs 765 GPT-4o tokens, 9.2x, all of it the assumed price ratio (dexa_platform/gateway/pricing.py; dexa_platform/README.md 'Why it's cheaper'; evals/RESULTS.md table; computation run in this session). Serving cost has never been measured in dollars.
- **"$0.02-0.06 / 1k pages" is a hard-coded string** in evals/modal_incumbent_docvqa.py with no recorded derivation. Deriving it: 30.2 pages/s = 108,720 pages/hr; at the repo's only GPU rate assumption, $1.80/hr (evals/stateful_cost_model.py, evals/RESULTS.md line 646), that is $0.017 per 1K pages at 100% utilization and 32-token outputs; at $2.50/hr (Tensormesh's listed reserved H200 rate, 05-research-kvcache.md) it is $0.023; the 1536px point gives $0.044-0.061. Realistic utilization, serving mode with CUDA graphs (the eval ran eager), and transcription-length outputs are unmeasured.
- **Qwen3-VL numbers are vendor-reported and cross-table.** Qwen3-VL-8B-Instruct DocVQA 96.1 is Table 4 of arXiv 2511.21631; GPT-5 (high) 91.5 is Table 2 of the same report, in the 235B comparison; the 8B table compares against GPT-5 nano at 88.2. Together's hosting page lists Qwen3-VL-32B at DocVQA 93.3 vs the report's 96.9 for the same checkpoint (05-research-docai.md). None of these are DocVQA-test ANLS reproduced on this harness.
- **Throughput is offline batch, not serving latency:** `llm.chat` over 200 prompts, eager mode, one image per prompt, max_tokens 32 (evals/modal_vlm_frontier.py). p50/p95 latency under concurrency is unmeasured.

### Unproven

| Claim | Experiment that proves or kills it | Est. cost |
|---|---|---|
| Content-aware in-model pruning beats resize at matched token budget (the moat) | Reimplement `get_rope_index` for a pruned token set in the HF Qwen2.5-VL path (the recorded blocker: [3,1671] positions vs [3,881] masked), then run evals/modal_vlm_moat.py at FRACTIONS 0.5/0.25; proceed if best prune minus resize >= +0.02 (the script's own verdict line) | 6-15 A100-hours; 5-10 engineering days (estimate) |
| Qwen3-VL-8B reproduces >= 93 ANLS on this harness and holds the resize frontier | Rerun modal_vlm_frontier.py with Qwen3-VL-8B, ANLS scorer, N >= 1,000 | 4-8 A100-hours; 2 days |
| The 0.925 vs 0.880 gap survives ANLS and N >= 1,000 | Re-score archived predictions with ANLS; rerun both models on 1,000 pages | 2 A100-hours plus OpenAI spend; 2 days |
| Adaptive per-page budgeting beats a fixed 1024px point | Route by page pixel density / text size to 768/1024/1536; measure accuracy and pages/GPU-hour | 4 A100-hours; 3 days |
| Measured $/1K pages at realistic utilization, serving mode, transcription outputs | `vllm bench serve`-style load at 8/32/128 concurrency; record GPU-hours and p95; repeat with 1,000-token outputs | 6 A100-hours; 3 days |
| gpt-5.6-luna / Gemini 3.8 Flash medium / Claude Haiku 4.5 accuracy on the same pages | Extend modal_incumbent_docvqa.py; record usage-based $/1K pages | API spend only; 1 day |
| Buyers move for a per-page price and a benchmark | 5 design partners rerun the harness on their pages; count conversions | 0 GPU-hours; 4 weeks |

## 7. MVP and 6-week build plan

**What ships first:** an OpenAI-compatible `/v1/chat/completions` endpoint (model name `doc-vlm`), a batch endpoint for PDFs, per-page metering with API keys, and a public benchmark page with the harness.

- **Week 1 - benchmark integrity.** Add ANLS to evals/modal_vlm_frontier.py; run Qwen2.5-VL-7B and Qwen3-VL-8B on >= 1,000 DocVQA pages in serving mode (CUDA graphs on) and log GPU-hours. Reuse: evals/modal_vlm_frontier.py, the `dexa-hf-cache` volume pattern. New: ANLS scorer, GPU-hour logger.
- **Week 2 - incumbents on the same pages.** Extend evals/modal_incumbent_docvqa.py to gpt-5.4, gpt-5.6-luna, Gemini 3.8 Flash (media_resolution medium), Claude Haiku 4.5, Azure DI Read+Layout, Google Document AI, Textract Queries, Reducto r-1, Mistral OCR 4.1; record returned usage, not list arithmetic. New: five vendor adapters.
- **Week 3 - endpoint hardening.** Reuse serve/modal_doc_vlm_serve.py, dexa_platform/gateway/{app,tenants,store}.py, dexa_platform/control (keys, metering). Fix pricing.py's Qwen token formula to patch 14 / merge 2 and add a per-page meter; remove the forced `stream=False` at app.py:127 and add SSE passthrough; include control/ in the Dockerfile (it copies only gateway/ and dashboard/ today; 03-build-inventory.md).
- **Week 4 - PDF and batch.** New: rasterizer, page budget policy encoding the frontier, batch job queue with a standby/off-peak price (Morph's Standby tier bills 50% when fleet capacity is under ~25%; DeepSeek off-peak is half price; 05-research-morph.md, 05-research-caching.md). Reuse dexa_platform/dashboard/index.html for a usage view.
- **Week 5 - publish.** Benchmark page plus harness release; design-partner onboarding with their pages. Reuse serve/demo_client.py as the quickstart.
- **Week 6 - time-boxed moat spike.** Pruned-grid mRoPE positions in the HF path (or a model without the coupling); run the moat test against its pre-registered gate; write the result either way.

## 8. Pricing model

Bill per page, by output type (QA/extraction vs full transcription), with a batch/standby tier, because that is the unit every incumbent bill already uses (Azure, Google, and AWS per 1K pages; Reducto, Unstructured, and Extend per page or credit; 05-research-docai.md). The architecture makes a per-page price expressible: at a fixed budget a page is a bounded token count (1,017 prompt tokens at 1024px), so pages per GPU-hour is a measurable constant (30.2 pages/s offline batch, eager, 32-token outputs) and cost per page is a derivation. A per-token provider cannot quote a page because the count depends on the client's resolution and the provider's tiling, and its revenue falls when it cuts tokens; per-page pricing keeps the 2.7x configuration gain on the provider's side. Modeled compute floor (assumptions stated, unmeasured in production): $0.017-0.023 per 1K pages at 100% utilization at $1.80-2.50/GPU-hour for QA-length outputs; transcription outputs unmeasured. Anchors: Azure Read $1.50/1K; Reducto r-1 $10/1K; Unstructured $15/1K; Gemini 3.8 Flash medium PDF page about $0.42/1K input-only and gpt-5.6-luna about $0.25/1K for a 1024x1024 image (both derived in 05-research-docai.md); Qwen3-VL-8B hosted per-token at $0.20/M (secondary aggregator) times 1,017 tokens is about $0.20/1K pages. Where the list price sits between the compute floor and those anchors is a founder decision.

## 9. Competitive facts

| Who | What they ship that is adjacent | What they do not ship (per the research files) | Source |
|---|---|---|---|
| Azure DI / Google Document AI / AWS Textract | Per-page OCR ($1.50/1K), layout ($10/1K), custom extraction ($30/1K), forms ($50/1K Textract) | An open-weight model, a same-pages benchmark against VLMs, per-token option | 05-research-docai.md |
| Reducto | r-1 parsing model at $0.01/page (2026-09-01), claims 20% lower error vs legacy pipelines; throughput tiers | Model size, latency, self-hosting details (undisclosed) | https://reducto.ai/blog/parse-r-1-model |
| LlamaParse / Extend / Unstructured | Credit or per-page parsing ($1.25 per 1K credits; $0.025/page; $0.015/page) | Disclosed underlying VLM (Unstructured 'VLM' strategy model unknown) | 05-research-docai.md |
| Mistral | OCR 4.1 at $4/1K ($2 batch), self-reported olmOCR-Bench 85.20; self-hosted for enterprise | Open weights for OCR 4 | https://mistral.ai/news/ocr-4/ |
| OpenAI / Anthropic / Google | Frontier VLMs with published image-token rules; Gemini native PDF text uncharged | Per-page pricing; open weights | 05-research-docai.md |
| Together / DeepInfra / Fireworks | Host Qwen3-VL (32B at $0.50/$1.50; 235B at $0.20/$0.88; 8B at $0.20/M per aggregator) | Per-page pricing; document-tuned operating points; same-pages benchmark | 05-research-docai.md |
| Qwen3-VL / InternVL3.5 / DeepSeek-OCR / olmOCR-2 / dots.mocr | Open weights; InternVL3.5's ViR halves visual tokens in-model; DeepSeek-OCR reports 97% precision under 10x compression; olmOCR-2-7B is a fine-tune of Qwen2.5-VL-7B | A hosted per-page service (no primary hosted listing found for the last four) | 05-research-docai.md |
| vLLM / SGLang | Video-token pruning (EVS, VidCom2); image-pruning RFC #45098 open | Image-token pruning; EVS incompatible with Qwen2.5-VL positional embeddings | 05-research-docai.md |
| Morph, Wafer (analogs) | Specialized small models on a custom stack with a standby tier (Morph); agent-tuned serving of open models across NVIDIA/AMD (Wafer) | Document AI products | 05-research-morph.md; 05-research-wafer.md |

## 10. Risks and pre-registered kill gates

| Risk | Measurement | Kills | Proceeds |
|---|---|---|---|
| Relaxed match flattered the GPT-4o gap | ANLS on the archived 200 pages, then 1,000 pages | Qwen minus GPT-4o < +1.0 pt ANLS | >= +2.0 pts, or parity with gpt-5.4-class at <= 1/5 measured $/page |
| Vendor-reported Qwen3-VL-8B does not reproduce | DocVQA ANLS on this harness | < 93.0 | >= 95.0 |
| The operating point is a config flag anyone sets | Pages/GPU-hour vs vanilla vLLM default on the same GPU | Advantage < 1.5x | >= 2.5x at <= 1 pt loss (measured 2.7x at -1.0 pt relaxed) |
| The moat sprint fails | Best prune minus resize at 25% tokens (modal_vlm_moat.py verdict) | <= 0.00 | >= +0.02 (script threshold) |
| Serving cost is not what the model says | Measured $/1K pages at 50% utilization, serving mode | > $0.42/1K for QA (Gemini medium-page anchor) | <= $0.10/1K |
| Frontier cheap tiers price-match | gpt-5.6-luna and Gemini 3.8 Flash accuracy and usage-based $/page on the same pages | Either within 1 pt ANLS at <= 2x our measured price | Both >= 3 pts below or >= 5x our price |
| Buyers do not move for a benchmark | Design-partner conversions after rerunning on their pages | 0 of 5 | >= 2 of 5 |

## 11. Founder decisions

- **Market size and whether the per-page market is worth entering.** Options: enter, or treat this as a feature of another direction. Cloud tier breakpoints (Azure $0.60/1K above 1M pages; Google above 5M) show volume exists but give no volumes; design-partner conversions would.
- **Whether Reducto r-1 at $0.01/page, Mistral OCR 4 at $2-4/1K, and the cloud OCR tiers are adjacencies that matter.** None publishes a same-pages result against open VLMs.
- **Hosted vs BYOC vs harness-only open source.** The repo has a hosted Modal endpoint and a BYOC compose file; the harness could ship alone. Evidence: whether design partners want the endpoint or the config.
- **Model choice.** Qwen2.5-VL-7B (measured 0.925 vs 0.880, relaxed, 200 pages) vs Qwen3-VL-8B (vendor 96.1, unreproduced) vs a document fine-tune (olmOCR-2-7B is a Qwen2.5-VL-7B fine-tune at 82.4 olmOCR-Bench; its synthetic training data cost about $0.12/page via Claude Sonnet 4). Week-1 ANLS runs inform it.
- **GPU class.** All measurements are A100-80GB; H100, L40S, and MI355X are unmeasured for this workload (MI355X rental $2.29-2.95/GPU-hour per Wafer; 05-research-wafer.md).
- **Moat sprint before or after market validation.** The repo recorded "gate the moat sprint on market validation" (evals/RESULTS.md); the alternative is the 6-15 GPU-hour spike first so the pitch is not "a config flag".
- **Product shape: QA/extraction vs full-page transcription vs both.** Only QA is measured.
- **Per-page vs per-token pricing, and whether to publish the compute floor.**

## 12. Combinations

- **Computer-use / screenshot VLM.** The proven screen-redundancy result (86% of 28px patches unchanged per action; realized reuse bounded at about 2-2.3x) uses the same Qwen2.5-VL encoder and pricing.py; one endpoint could serve pages and screenshots (01-evidence-ledger-dexa.md, cua-screen-redundancy, delta-tolerance-sweep).
- **Session-stateful serving.** Multi-question workflows over one document reuse a page prefix; vLLM prefix-cache hits measured 12x at 4k and 25-34x at 16k TTFT on a text model (01-evidence-ledger-dexa.md, vllm-warmstart-prefix-cache); page-image prefix hits are unmeasured.
- **Morph-style specialization.** Document fine-tune plus task-specific speculator is the playbook Morph and Relace describe for code (05-research-morph.md); olmOCR-2 shows the fine-tune half exists for this base.
- **Wafer-style hardware optimization.** Tuned engine and kernel configs on AMD parts could lower the compute floor; unmeasured for VLM prefill.
- **Benchmark-as-product.** The harness run publicly across vendors on the same pages is a standalone asset whichever serving direction wins.
- **Batch/standby capacity sharing with voice or agent directions.** Page backlogs fill idle capacity; Morph's 50%-price standby tier is the precedent (05-research-morph.md).