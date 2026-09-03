# Candidate I — "Fast Parse": the Morph playbook on document-to-structure

Morph's playbook: a narrow, repetitive agent workload whose output is highly predictable ("roughly 70 or 80% of the content is almost exactly the same"), a small specialized model plus a task-shaped speculator, and "almost make our own inference engine just for this task" (05-research-morph.md). **Neither ledger contains a model-training result, a full-page parse measurement, or a speculative-decoding measurement; every in-repo number below is a short-answer (max_tokens=32), single-run, 200-page result on one A100.**

## 1. Thesis

Fast Parse is a one-model inference company: it serves a single narrow, high-volume task — turn a document page image into structured output (page text as markdown plus schema-conforming JSON fields) — on a purpose-built engine, the way Morph serves Fast Apply. The task is chosen because its output is largely a copy of text visible in the input (the property Morph exploits with a task-shaped speculator — unmeasured for documents); because it is the one direction where the repos hold a measured model-side result (an open 7B VLM at a tuned visual-token budget scored 0.925 vs GPT-4o's 0.880 on 200 short-answer DocVQA pages, at 2.7x the throughput of the same model at higher resolution); and because public benchmarks, Apache-2.0 base models, and a published synthetic-data recipe exist. The sentence a customer would repeat, if the benchmark holds: "It parses a page as accurately as the frontier models, in a fraction of the time, and I pay per page." The comparator actually measured is GPT-4o, a 2024 model; current frontier models are unmeasured on the same pages.

## 2. Customer and workload

**Who buys.** Engineering teams doing document intake (invoices, claims, KYC, lending, contracts), RAG ingestion, and agent tools that read PDFs. They already pay three vendor classes (05-research-docai.md unless noted):
- Cloud APIs, per 1,000 pages: Azure Document Intelligence Read $1.50, Layout/Prebuilt $10, Custom $30; Google Document AI OCR $1.50, Form Parser/Custom Extractor $30; Textract DetectDocumentText $1.50, Tables $15, Queries $15, Forms $50.
- Startups: Reducto r-1 $0.01/page (launched 2026-09-01; its legacy agentic pipelines cost 3-6 cents/page); Unstructured $0.015/page after 10,000 free; Mistral OCR 4.1 $4/1K (batch half price).
- Frontier VLMs per page: Claude 1,296 tokens for 1000x1000 (about $1.30 per 1,000 pages on Haiku 4.5); GPT-5.4 1,229 tokens for 1024x1024 (about $0.0031/page); Gemini 560 tokens per PDF page at medium resolution (about $0.42 per 1,000 pages on 3.8 Flash, derived).
- Self-hosters: vLLM/SGLang serving Qwen3-VL-8B (DocVQA 96.1 / OCRBench 896 vs GPT-5 high 91.5 / 810 — Qwen-reported, not independently replicated) or olmOCR-2-7B / dots.mocr 3B (in vLLM 0.11.0).

**Workload shape (assumed; no document-workload telemetry exists in the research files).** Model 2B-8B. Context = one page image (1,017 visual tokens at 1024 px, 2,201 at 1536 px, measured for Qwen2.5-VL-7B) plus a shared instruction/schema prefix. Output = a few tokens (field QA) to hundreds (full-page markdown). Stateless, single-shot per page; batch vs interactive mix unmeasured.

## 3. The pain, in the customer's words

Paraphrases assembled from the research files; no customer interviews exist in the repos.

- "The open 8B models report beating GPT-5 on DocVQA, but nobody serves them tuned for pages. vLLM only has `--video-pruning-rate`; the image-pruning RFC (#45098, opened 2026-06-10) had no maintainer comment at fetch time."
- "Frontier accuracy is fine; the bill is the page images — 1,229 tokens a page."
- "Agentic parsers are accurate and slow, and nobody publishes seconds per page. I want one number per page, a latency I can put in an agent loop, and a benchmark I can rerun."

## 4. Value proposition and the proof-of-value benchmark

**Claim to prove.** A specialized 2B-8B document model on a purpose-built engine delivers accuracy at or above the best open OCR models and the frontier APIs on full-page parse, at several-x the pages per GPU-second of a *well-configured* vLLM/SGLang serving the same base model, and several-x lower single-page latency, priced per page.

**Metrics.** (A) Batch: pages per GPU-second at a fixed accuracy floor, converted to $/1,000 pages at a stated GPU rate. (B) Interactive: p50/p95 seconds per page single-stream. (C) Accuracy: olmOCR-Bench (parse), DocVQA (fields), plus a schema-extraction eval to be built. No in-repo number exists for (A) or (B) on full-page output.

**Setup.** Same pages, same GPU (A100-80GB first; H100 second), same base model, greedy, three seeds; harness generalizes `/home/user/dexa/evals/modal_vlm_frontier.py` and `modal_incumbent_docvqa.py` (200 DocVQA pages, max_tokens=32).

**Baselines, named.** (1) *Well-configured* vanilla vLLM, Qwen3-VL-8B: the same `max_pixels` as our budgeter, prefix caching on, FP8 weights (Relace converts its apply model to FP8 via llm-compressor), and vLLM's built-in n-gram/prompt-lookup speculative method with the OCR text in the prompt — if it accepts Qwen3-VL image inputs (not covered by the research files; week-1 check). The ledger's 2.7x is a resize "a client can apply before any API", so a default-resolution baseline would be a strawman. (2) SGLang, same. (3) Hosted Qwen3-VL-8B ($0.20/M on DeepInfra/Fireworks, aggregator). (4) Frontier, current: GPT-5.4, Gemini 3.x Flash, Claude 5 — plus GPT-4o (0.880 on the same pages; output unrecorded except that number). (5) Specialists: Reducto r-1, Mistral OCR 4 (self-reported olmOCR-Bench 85.20), olmOCR-2-7B (82.4), dots.mocr (83.9), Nanonets OCR-3 (87.4, medium confidence).

**Pre-registered targets (chosen thresholds, not measurements).** Batch: >= 3x pages/GPU-second vs baseline (1) at <= 1 pt loss on both benchmarks; the 3x must come from speculation, pruning, scheduling, and training. Interactive: >= 2x single-stream latency reduction from the trained copy-speculator vs baseline (1)'s n-gram draft (Morph reports 3.07x vs 1.93x for a task-trained draft over a generic one — on code). Accuracy: >= 85.2 on olmOCR-Bench and >= base model on DocVQA.

**Why a skeptical buyer believes it.** A rerunnable public harness, accuracy-vs-budget curves, three seeds per number (both repos are single-seed for every headline, 04-conflicts.md), and third-party listing — OpenRouter shows 1,537 tps observed for Morph V3 Fast against Morph's 10,500 tok/s claim.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| Base model Qwen3-VL-8B (or 4B/2B) | OSS, Apache-2.0 | Page understanding; DocVQA 96.1 / 95.3 / 93.3 (Qwen-reported) |
| Task fine-tune (LoRA SFT on synthetic pairs) | Custom | Format and schema following; peer reference Relace (3-8B base, ~145k examples, LoRA rank 128, one H200); unproven for documents |
| Copy speculator | Custom | Draft = OCR/PDF text via n-gram lookup plus a trained draft head; VLM verifies. N-gram drafts are off the shelf (Baseten NGram/Eagle/MTP; Together ATLAS); the moat is the trained head. vLLM multimodal speculative support unverified |
| Visual-token budgeter | Custom on vLLM | Resize to the measured 1024 px point; content-aware pruning blocked by Qwen2.5-VL mRoPE (01-ledger `vlm-moat-mrope`); Qwen3-VL unmeasured |
| Instruction-prefix cache | vLLM automatic prefix caching | Only pre-image tokens form a shared prefix; a schema after the image is not cacheable. Measured ~12x TTFT on a 4k *text* prefix (280 -> 24 ms); a few-hundred-token prefix saves milliseconds |
| Page scheduler | Custom on vLLM | Batch and latency lanes; pattern from `/home/user/dexa/evals/modal_verifier_sched.py` (drives `LLMEngine` add_request/step/abort) |
| Gateway + metering | Reused: `/home/user/dexa/dexa_platform/{gateway,control}` | OpenAI-compatible API, image-token accounting (`pricing.py`), keys, usage rollups; 29 tests pass (7 are session tests); Dockerfile omits control/ |
| Load harness | Reused: `/home/user/voice-inference/vkv/{loadgen,metrics}` | N-slot closed-loop load, JSONL events, p95 gating; needs text/image prompts, aiohttp, server metrics, multi-seed |

Differentiation, if any, lives in the fine-tuned model, the trained draft head, the multimodal input path and the scheduler — all unmeasured. vLLM is kept for paged attention, batching, prefix caching and grammar decoding. vLLM vs SGLang vs a fork is a founder decision.

```
 page image + optional PDF text ──> Gateway (keys, per-page metering)
                                       │
                        Page scheduler (batch lane | agent lane)
            ┌──────────────────────────┼──────────────────────────┐
   Visual-token budgeter      Instruction-prefix cache      Copy speculator
   (resize -> ~1k tokens;     (pre-image tokens only)       (OCR/PDF text -> draft;
    pruning: re-attempt)                                      VLM verifies)
            └──────────────────────────┼──────────────────────────┘
                    vLLM core: Qwen3-VL-8B + LoRA, constrained JSON decode
                                       │
                         markdown + JSON fields, confidence, tokens
```

## 6. Evidence

### Proven
- Visual-token budget lever (Qwen2.5-VL-7B, vLLM 0.24.0, A100-80GB, 200 DocVQA validation pages, relaxed match): 1536 px 0.935 acc / 11.4 img/s / 2,201 tokens; 1024 px 0.925 / 30.2 img/s / 1,017 tokens (2.7x, -1.0 pt); 768 px 0.885 / 47.2 (4.1x, -5 pts); 384 px 0.445. Source: `/home/user/dexa/evals/RESULTS.md` 'Multimodal execution thesis'; 01-ledger `doc-vlm-frontier`. Caveats: `max_tokens=32`; one offline `LLM.chat` batch, not a served endpoint; one run, one GPU, no seeds; naive resize that "sizes the prize but isn't yet the moat".
- vLLM prefix caching, in-engine: cold vs warm TTFT ~280 -> ~24 ms at 4k, ~1,250 -> ~45 ms at 16k (Qwen2.5-7B text, in-GPU hit; `evals/RESULTS.md`).
- Decode-step reference: n=1 ~22 ms/step, flat across context (Llama-3.1-8B, vLLM 0.24.0, A100, eager, prefix caching off, gen=64, one run; 01-ledger `vllm-decode-pathology`). Argues only that a hundreds-of-token parse is decode-dominated single-stream; the VLM's split is unmeasured.

### Bounded / contradicted
- Accuracy parity vs GPT-4o: 0.925 vs 0.880 on the same 200 pages. Ledger status BOUNDED (01-ledger `gpt4o-docvqa-head-to-head`): GPT-4o run output unrecorded except that number in `docs/FINDINGS.md`; GPT-4o-mini accuracy unrecorded; GPT-4o is a 2024 model; no current frontier model was run.
- "~40x cheaper than GPT-4o" = 3.4x fewer tokens (1,105 vs 322 for a 1280x800 screenshot, formula) x 12.5x assumed price ($0.20/M vs $2.50/M); accuracy was measured at 1,017 tokens, not 322; the Qwen cost row "$0.02-0.06/1k pages" is hardcoded; GPT-4o's $/page is unrecorded (04-conflicts.md conflict 10).
- Neither repo doc endorses this direction: `evals/RESULTS.md` (Aug 2) calls every inference-stack path closed for a small team; `docs/FINDINGS.md` (Sep 2) calls the VLM result "a real economics/config win, not a novel-kernel moat. Copyable."
- Content-aware pruning is OPEN, not failed: Qwen2.5-VL `get_rope_index` expects the full grid ([3,1671] computed vs [3,881] masked); no accuracy numbers.

### Unproven

| claim | experiment that proves or kills it | est. cost |
|---|---|---|
| Output is mostly a copy of input text | 1,000 pages with Qwen3-VL-8B; share of output tokens verbatim in PDF text / CPU OCR, scanned vs native | ~2 GPU-h, 2 days |
| vLLM speculative decoding accepts Qwen3-VL images; a well-configured baseline does not erase the gap | Enable n-gram; acceptances > 0 with identical greedy output; baseline (1) vs default vLLM at equal accuracy | ~4 GPU-h, 3 days |
| Trained copy-speculator >= 2x single-stream over n-gram | Trained draft head vs n-gram vs none; acceptance length, s/page | ~20 GPU-h, 2 weeks |
| Prefill/decode split, full-page throughput and $/page, replicated | Time prefill and decode at 1024 px; pages/GPU-s at 300-800 output tokens over HTTP, N = 8..256, 3 seeds x {A100, H100} | ~27 GPU-h, 1 week |
| Current frontier accuracy on the same 200 pages | Rerun `modal_incumbent_docvqa.py` with GPT-5.4, Gemini 3.x Flash, Claude Sonnet 5 | ~$100 API, 1 day |
| LoRA-SFT 8B/4B beats base and reaches >= 85.2 olmOCR-Bench; a trusted extraction eval exists | ~50k synthetic pairs at ~$0.12/page (olmOCR recipe, Claude Sonnet 4 teacher; ~$6k); LoRA SFT on one H100/H200; olmOCR-Bench, DocVQA, held-out extraction from public KIE sets | ~40 GPU-h + ~$6k, 3 weeks |
| Content-aware pruning beats resize on Qwen3-VL | Check Qwen3-VL positions; re-implement masked positions; sweep 0.5/0.25 vs resize | ~10 GPU-h, 2 weeks |

## 7. MVP and 6-week build plan

**Ships first:** a hosted per-page `/v1/parse` endpoint (page or PDF in; markdown + JSON fields out) with a public, rerunnable benchmark. Model v0 is stock Qwen3-VL-8B on the specialized engine; the fine-tune is a parallel track.

- **Week 1 — baselines and eval.** Port the two eval scripts to Qwen3-VL-8B and olmOCR-Bench; build baseline (1); run baselines (1)-(5); measure copy fraction and prefill/decode split at full-page lengths. Gates 1-3, 7. Start design-partner outreach now; no customer conversations exist in the repos.
- **Week 2 — copy speculator.** OCR/PDF-text draft inside vLLM; acceptance and s/page vs baseline (1). If vLLM's speculative path rejects image inputs, this becomes engine work of unknown length (slip risk A). Gate 4.
- **Week 3 — engine v0.** Budgeter at 1024 px; constrained JSON; instruction-prefix caching; scheduler skeleton from `modal_verifier_sched.py` (needs a new HTTP serving loop — slip risk B). Wire into `dexa_platform/gateway` (drop `stream=False` at `app.py:127`; per-page metering in `pricing.py`; add control/ to the Dockerfile); deploy from `serve/modal_doc_vlm_serve.py`.
- **Week 4 — throughput and pricing.** Adapt `voice-inference/vkv/loadgen` + `metrics` (text/image prompts, tokenizer, aiohttp, server-metric scraping); sweep concurrency on A100 and H100; derive $/1k pages; publish the benchmark page. Gates 5, 8.
- **Weeks 4-6 (parallel) — data and fine-tune.** Synthetic pairs via the olmOCR recipe; LoRA SFT 8B and 4B; eval. Gate 6. Budget three weeks, not the draft's one.
- **Week 6 — pilots.** BYOC container (extend `dexa_platform/docker-compose.byoc.yml`, which already runs `vllm/vllm-openai:v0.24.0` with Qwen2.5-VL-7B); design partners. Gate 9 needs 4+ weeks of pilot logs beyond week 6.

**Realistic calendar.** First credible public proof of the engine claim (baseline (1) vs engine v0 on full-page parse, three seeds, two GPUs, public harness): about 8-10 weeks with 1-2 engineers if vLLM's speculative path accepts multimodal inputs; 14-18 weeks if it must be built or if the fine-tuned model is part of the proof.

Reused: the two eval scripts, `serve/modal_doc_vlm_serve.py`, `dexa_platform/{gateway,control}`, `docker-compose.byoc.yml`, the `modal_verifier_sched.py` pattern, voice-inference loadgen/metrics. New: speculator, budgeter/pruner, scheduler policy, data pipeline, fine-tune, benchmark page. Not reused: KV connectors, tiering, cartridges, compaction.

## 8. Pricing model

Per page, two SKUs (parse; parse + extract with schema), volume tiers, batch discount, ZDR option — the units buyers already use. A per-page price is expressible because cost per page is bounded by design: the budgeter fixes visual tokens near 1,017, the instruction prefix is cached, and output is bounded by the page. A per-token provider cannot quote a page price because image tokens vary with resolution and tokenizer rules.

Illustrative floor, derived and conditional: 30.2 pages/s at 1024 px (short answers, offline batch) is 108,720 pages/GPU-hour; at Modal's A100-80GB rate of $2.50/hr (05-research-gpu-pricing.md) that is about $0.023 per 1,000 pages of field QA. Full-page parse emits hundreds of tokens: at ~22 ms/step single-stream, 500 output tokens is ~11 s per page before speculation, so the batch lane depends on decode batching that is unmeasured. Margin against $10-$50/1K list prices is what the week-4 benchmark must confirm.

## 9. Competitive facts

| who | adjacent thing shipped | not shipped (per research files) | source |
|---|---|---|---|
| Morph | Fast Apply 10,500 tok/s claimed (custom kernels + task speculator; OpenRouter observes 1,537 tps); Compact 33k tok/s; Reflexes ("forked from vLLM"); Glance; open-model hosting | No document parsing/OCR product listed | 05-research-morph.md |
| Reducto | r-1 parse $0.01/page (2026-09-01); Extract $20/1K; throughput guarantees 200/350/500+ concurrent pages | Model size, latency, self-hosting undisclosed | 05-research-docai.md |
| Mistral | OCR 4.1 $4/1K (batch half price); self-reported olmOCR-Bench 85.20; enterprise self-host | Per-page latency unpublished | 05-research-docai.md |
| olmOCR-2 / dots.mocr / Chandra / Nanonets | Open 7B (82.4), 3B MIT (83.9), Chandra (83.1), Nanonets OCR-3 (87.4, medium confidence); dots in vLLM 0.11.0; olmOCR data+code released | No hosted endpoint found for the open ones | 05-research-docai.md |
| DeepSeek-OCR / InternVL3.5 | Model-level visual-token compression: 97% precision at <10x compression, <800 vision tokens beating MinerU2.0, 200k+ pages/day on one A100-40G; ViR ~50% fewer tokens | No hosted API found | 05-research-docai.md |
| Azure / Google / AWS | Per-page APIs $1.50-$50 per 1K; commitment tiers; Azure disconnected-container tiers (Read 16000K $77,760/year); Google custom-processor hosting | No open weights; AWS self-host option not found | 05-research-docai.md |
| Together / DeepInfra / Fireworks / Baseten | Hosted Qwen3-VL per token (Together 32B; DeepInfra 235B; 8B via aggregator); in-engine speculation (Together ATLAS; Baseten NGram/Eagle/MTP); Fireworks: >95% of tokens are customer-specialized models | No per-page SKU, no page-tuned engine found | 05-research-docai.md, 05-research-providers.md |
| vLLM / SGLang | Video-token pruning (EVS); prefix caching; grammar decoding | Image-token pruning (RFC #45098 open); SGLang EVS incompatible with Qwen2.5-VL positions | 05-research-docai.md |

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| 1. Output is not copy-like | Copy fraction on 1,000 pages, scanned vs native (wk 1) | < 40% verbatim | >= 70% |
| 2. Well-configured baseline erases the lever | Baseline (1) vs default vLLM at equal accuracy (wk 1) | baseline (1) captures >= 80% of our week-4 gain | our gain >= 2x over baseline (1) |
| 3. Parse is prefill-bound and pruning stays blocked on Qwen3-VL | Prefill/decode split at 200-500 output tokens (wk 1); matched-budget pruning vs resize | decode < 30% of page time and no pruning gain at 0.5 after 2 weeks | decode >= 50% or >= +1 pt at 0.5 |
| 4. Speculation does not pay, or vLLM cannot speculate on VLM inputs | Trained draft vs n-gram vs none; acceptances > 0 on image inputs (wk 2) | < 1.5x over n-gram, or > 3 weeks to make the path work | >= 2x |
| 5. Throughput claim fails under load | N = 8..256, p95 s/page, 3 seeds, A100 + H100 (wk 4) | < 2x vs baseline (1) at equal accuracy | >= 3x |
| 6. Fine-tune does not beat base / specialists | olmOCR-Bench, DocVQA, held-out extraction (wk 4-6) | < base or < 82.4 | >= 85.2 and >= base |
| 7. Parity fails vs current frontier | GPT-5.4 / Gemini 3.x / Claude 5 on the same 200 pages and olmOCR-Bench | trail by > 3 pts on both | within 1 pt or ahead on either |
| 8. Unit economics below list | Measured full-parse pages/GPU-h x GPU rate vs $10/1K | COGS > $5/1K | COGS <= $1/1K |
| 9. Buyers ignore page latency | Pilot logs, interactive vs batch share | 0 of 3 pilots use the agent lane | >= 2 of 3 |
| 10. Base-model shelf life (token-efficient architectures; "fast apply is dead") | Open-model and frontier $/page, s/page tracked quarterly | a stock open model matches our s/page within 1.5x | gap >= 3x |

## 11. Founder decisions

- **Task choice.** The draft picked document parse from seven Morph-style candidates: document parse (the only in-repo model-side evidence; public benchmarks; Apache-2.0 bases); voice turn model (serving evidence only — 287,015 archived turns, arm D 1.5x at one seed on an L40S with synthetic traces; LiveKit serves Gemma 4 31B on SGLang with speculative decoding at 192 ms TTFT); context compaction (Morph Compact 33k tok/s; no trace corpus in repo); agent-trace classifiers; code-search subagent; screenshot-diff perception (capturable ceiling 2.34x); tool-call routing. Options: this task, another row, or two sharing the speculator. Evidence: week-1 copy fraction and baselines.
- **Engine base.** vLLM (ledger-pinned 0.24.0; grammar and speculative paths built in) vs SGLang (better at branched decode per the ledger's SGLang gate; EVS incompatible with Qwen2.5-VL positions) vs a fork. Evidence: week-1 multimodal speculative-decoding check on each.
- **Market size and incumbents' adjacency.** Neither judged. Reducto r-1 at $0.01/page, Mistral OCR 4, open olmOCR/dots/Chandra/Nanonets, DeepSeek-OCR's token-efficient architecture, and Morph's ability to add a doc model are facts, not verdicts; no volume disclosures exist in the research files. Options: compete on latency + per-page price with a public benchmark, or pick another task. Evidence: week-1 baselines.
- **Hosted vs BYOC vs open weights.** Gateway and compose file support hosted and BYOC today; Mistral self-hosts for enterprise; Azure sells disconnected containers; olmOCR/dots are open; Morph does not release weights. Evidence: pilot procurement asks.
- **Synthetic-data teacher.** The olmOCR recipe uses Claude Sonnet 4 at ~$0.12/page; training on a frontier model's outputs is a terms-of-service decision and the ~$6k assumes 50k pages. Options: frontier teacher, open teacher (Qwen3-VL-235B), human-labeled subset. Evidence: accuracy per data source.
- **Speculator form, accuracy metric, publishing.** Prompt-lookup only vs trained draft head (Morph: 3.07x vs 1.93x, code); relaxed-match DocVQA on 200 validation pages vs ANLS on test vs olmOCR-Bench; publish the harness or not (Morph's benchmark pages returned HTTP 429 in research). Evidence: week-2 acceptance lengths; pilot asks; pilot conversion.
- **Model size, GPU class, parse-first vs extract-first, pricing unit, free tier, kill-gate thresholds, team size.** Qwen3-VL-8B (96.1) vs 4B (95.3) vs 2B (93.3) vs dots.mocr/olmOCR bases; A100-80GB $1.39-$2.79/hr vs H100 $3.29-$3.99 vs MI355X $2.29-$2.95 (05-research-gpu-pricing.md, Wafer); parse (benchmarks, higher copy fraction) vs extract ($20-$50/1K); per page vs credit vs token; 10,000 free pages; every threshold in Section 10; one or two engineers. All chosen by the draft; each needs ratification.

## 12. Combinations

- **Document/CUA gateway direction (dexa_platform):** Fast Parse is the first model behind that gateway; `pricing.py`'s image-token accounting becomes the buyer-facing savings meter.
- **Verifier-guided search direction:** for hard pages, best-of-N with a schema/consistency verifier and early stop (2.63x fewer tokens at equal pass@16 on HumanEval); unmeasured on documents.
- **Stateful-session direction:** multi-page documents share an instruction prefix; vLLM prefix caching covers it; no connector is needed.
- **Morph-catalog expansion:** if the playbook works once, compaction and tool routing reuse the speculator and scheduler; each needs its own data and eval, absent from the repos.
