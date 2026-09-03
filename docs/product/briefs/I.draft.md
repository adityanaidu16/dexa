# Candidate I — "Fast Parse": the Morph playbook on document-to-structure

### Task selection

Morph's playbook, per its founder: pick a narrow, repetitive agent workload whose output is highly predictable ("roughly 70 or 80% of the content is almost exactly the same... you're essentially using the original code as a guess"), train a small specialized model plus a task-shaped speculator, and "almost make our own inference engine just for this task" (05-research-morph.md, Infra Pod transcript). Candidates were rated on (a) output predictability, (b) ledger evidence, (c) data/eval availability, (d) stack-specialization room. Ratings are qualitative. **Neither ledger contains any model-training result; the model side of every candidate is unproven.**

| candidate | (a) predictability | (b) ledger evidence | (c) data / eval | (d) stack specialization |
|---|---|---|---|---|
| **Document page -> structured text + fields** | High: output is largely verbatim text from the page; an OCR text layer is a natural draft | Strongest model-side result in either repo: Qwen2.5-VL-7B 0.925 vs GPT-4o 0.880 on 200 DocVQA pages; 2.7x throughput at -1.0 pt from visual-token budget (01-ledger) | Public: DocVQA, OCRBench, OmniDocBench, olmOCR-Bench; Qwen3-VL-8B (Apache-2.0) DocVQA 96.1; olmOCR synthetic recipe ~$0.12/page | Visual-token budget/pruning (none for images in vLLM/SGLang), copy-speculation, constrained JSON, schema-prefix cache — **selected** |
| Voice turn model | Low-Med: replies generated, not copied | Strong serving evidence (287k turns; arm D 300 vs 200 sessions, 1.5x), zero model evidence (02-ledger) | No real transcripts; all synthetic | Session residency (1.5x measured); LiveKit already serves Gemma 4 31B with spec decoding at 192 ms TTFT |
| Context compaction (Compact analog) | Very high: output = input minus lines | dexa compaction is KV-level (AM vs H2O, BOUNDED), not text | Needs an agent-trace corpus; none in repo | Copy-speculation; Morph ships it at 33k tok/s |
| Agent-trace classifiers (Reflexes analog) | n/a (no decode) | None | Needs labeled traces | Decode-removed engine (Morph: 'forked from vLLM') |
| Code search subagent | Med: short tool calls | Indirect: verifier early-stop 2.63x fewer tokens (HumanEval) | SWE-bench; needs RL (Relace: 8xH200 ~1.5 days) | Parallel tool-call scheduling; three shipped peers |
| Screenshot-diff perception (CUA) | High for actions | 86% patches unchanged but capturable ceiling ~2.3x; mRoPE blocker | OSWorld/WebArena; repo used a synthetic CRM | Delta encoding bounded ~2x |
| Tool-call routing (Router analog) | High: label output | None | Needs routing corpus | Prefill-only |

## 1. Thesis

Fast Parse is a one-model inference company: it serves a single narrow, high-volume task — turn a document page image into structured output (page text as markdown plus schema-conforming JSON fields) — on a purpose-built engine, the way Morph serves Fast Apply. The task is chosen because its output is overwhelmingly a copy of text already visible in the input, the property Morph exploits with a task-shaped speculator; because it is the one direction where the repos hold a measured model-side result (an open 7B VLM at a tuned visual-token budget matched GPT-4o on DocVQA at 2.7x the throughput); and because public benchmarks, Apache-2.0 base models, and a published synthetic-data recipe exist. Buyers are teams building document-intake pipelines and agents that read PDFs. The sentence a customer repeats: "It parses a page as accurately as the frontier models, in a fraction of the time, and I pay per page."

## 2. Customer and workload

**Who buys.** Engineering teams doing document intake (invoices, claims, KYC, lending, contracts), RAG ingestion, and agent tools that read PDFs. They already pay three vendor classes (05-research-docai.md unless noted).
- Cloud APIs, per 1,000 pages: Azure Document Intelligence Read $1.50, Layout/Prebuilt $10, Custom $30; Google Document AI OCR $1.50, Form Parser/Custom Extractor $30; Textract DetectDocumentText $1.50, Tables $15, Queries $15, Forms $50.
- Startups: Reducto r-1 $0.01/page (launched 2026-09-01; Reducto says its legacy agentic pipelines cost 3-6 cents/page); Unstructured $0.015/page; Extend Parse $0.025 / Extract $0.0375 per page; LlamaParse 1-45 credits/page at $1.25 per 1,000 credits; Mistral OCR 4.1 $4/1K ($2 batch).
- Frontier VLMs per image token: a 1000x1000 page is 1,296 tokens on Claude (about $1.30 per thousand on Haiku 4.5); 1024x1024 is 1,229 tokens on GPT-5.4 (about $0.0031); Gemini bills 560 tokens per PDF page at medium resolution (about $0.42 per 1,000 pages input on Gemini 3.8 Flash, derived in the research file).
- Self-hosters: vLLM/SGLang serving Qwen3-VL (8B: DocVQA 96.1 / OCRBench 896 vs GPT-5 high 91.5 / 810) or 3-7B OCR models (olmOCR-2-7B; dots.mocr 3B, in vLLM 0.11.0). Hosted: Together Qwen3-VL-32B $0.50/$1.50 per M; Qwen3-VL-8B at $0.20/M on DeepInfra/Fireworks (aggregator, medium confidence); Fireworks' 4B-16B tier $0.20/M (05-research-providers.md).

**Workload shape.** Model 2B-8B. Context per request = one page image (1,017 visual tokens at 1024 px, 2,201 at 1536 px, measured for Qwen2.5-VL-7B) plus a shared schema/instruction prefix. Output = a few tokens (field QA) to hundreds (full-page markdown). Turn pattern: stateless, single-shot per page. Concurrency: bursty batch backfills plus interactive agent reads. Idle pattern: none — throughput-shaped, not session-shaped.

## 3. The pain, in the customer's words

Paraphrases assembled from the research files; no customer interviews exist in the repos.

- "Every vendor prices a different unit — pages, credits, 32x32 patches times a multiplier, 768 px tiles, media_resolution levels. I can't compare quotes without my own eval."
- "The open 8B models beat GPT-5 on DocVQA, but nobody serves them tuned for pages. vLLM only has `--video-pruning-rate`; the image-pruning RFC (#45098, opened 2026-06-10) has no maintainer comment."
- "Frontier accuracy is fine; the bill is the page images — 1,229 tokens a page."
- "Agentic parsers are accurate and slow, and nobody publishes seconds per page. I want one number per page, a latency I can put in an agent loop, and a benchmark I can rerun."

## 4. Value proposition and the proof-of-value benchmark

**Claim to prove.** A specialized 2B-8B document model on a purpose-built engine delivers frontier-class page accuracy at several-x the pages per GPU-second of vanilla vLLM/SGLang serving the same base model, and several-x lower single-page latency, priced per page.

**Metrics.** (A) Batch: pages per GPU-second at a fixed accuracy floor, converted to $/1,000 pages at a stated GPU rate. (B) Interactive: p50/p95 seconds per page single-stream. (C) Accuracy: olmOCR-Bench (parse), DocVQA relaxed-match (fields), plus a schema-extraction eval to be built.

**Setup.** Same pages, same GPU (A100-80GB first, matching the ledger; H100/B200 later), same base model, greedy. Harness generalizes `/home/user/dexa/evals/modal_vlm_frontier.py` (five image budgets, 200 DocVQA pages, `max_tokens=32`) and `/home/user/dexa/evals/modal_incumbent_docvqa.py` (GPT-4o head-to-head, temperature 0).

**Baselines, named.** (1) vanilla vLLM, Qwen3-VL-8B, default resolution, no speculation; (2) SGLang, same; (3) hosted: DeepInfra/Fireworks Qwen3-VL-8B, Together Qwen3-VL-32B; (4) frontier: GPT-4o (0.880 measured on the same pages), Gemini Flash, Claude; (5) specialists: Reducto r-1, Mistral OCR 4 (self-reported olmOCR-Bench 85.20), olmOCR-2-7B (82.4), dots.mocr (83.9).

**Pre-registered targets (chosen thresholds, not measurements).** Batch: >= 3x pages/GPU-second vs baseline (1) at <= 1 pt loss on both benchmarks. The 2.7x measured from resize alone is the floor and is buyer-replicable, so the proprietary increment must come from speculation, pruning, and training. Interactive: >= 2x single-stream latency reduction from copy-speculation vs the same engine without it. Accuracy: >= 85.2 on olmOCR-Bench and >= base model on DocVQA.

**Why a skeptical buyer believes it.** Public benchmarks, a rerunnable harness, accuracy-vs-token-budget curves rather than one point, and third-party listing where throughput is observed independently — OpenRouter shows 1,537 tps observed for Morph V3 Fast against Morph's 10,500 tok/s claim; that is the check buyers run.

## 5. Architecture

| component | custom or OSS | role |
|---|---|---|
| Base model Qwen3-VL-8B (or 4B/2B) | OSS, Apache-2.0 | Page understanding; DocVQA 96.1 / 95.3 / 93.3 |
| Task fine-tune (LoRA SFT on synthetic parse+extract pairs) | Custom | Output format and schema following; Relace's apply recipe (3-8B base, ~145k examples, LoRA rank 128, one H200) is the peer reference |
| Copy speculator | Custom | Draft = OCR text layer / native PDF text in the prompt; n-gram prompt-lookup plus a trained draft head; the VLM verifies. N-gram drafts exist off the shelf (Baseten BIS-LLM lists NGram/Eagle/MTP), so the moat is the trained head and the OCR-draft construction, not lookup itself |
| Visual-token budgeter | Custom on vLLM | Per-page resize to the measured 1024 px point; content-aware pruning blocked by Qwen2.5-VL mRoPE (01-ledger `vlm-moat-mrope`), to be re-attempted on Qwen3-VL |
| Constrained JSON decoder | OSS grammar backend + custom schema compiler | Schema-valid fields; no retries |
| Schema-prefix cache | vLLM automatic prefix caching | Shared instruction/schema prefix hit per page (measured ~12x TTFT at 4k on a text model) |
| Page scheduler | Custom on vLLM | Batch lane for backfills, latency lane for agent reads; pattern from `/home/user/dexa/evals/modal_verifier_sched.py` (drives `LLMEngine` add_request/step/abort) |
| Gateway + metering | Reused: `/home/user/dexa/dexa_platform/{gateway,control}` | OpenAI-compatible API, image-token cost accounting (`pricing.py`), keys, usage rollups; 29 tests pass |
| Load harness | Reused: `/home/user/voice-inference/vkv/{loadgen,metrics}` | N-slot closed-loop load, JSONL events, p95 gating; needs text/image prompts and aiohttp |

Differentiation lives in the fine-tuned model, the copy speculator, and the engine's multimodal input path plus constrained decoding. vLLM is kept for paged attention, batching, and prefix caching; the image-token path, speculative proposer, and scheduler policy are replaced or added. Nothing session-stateful is required.

```
 page image + optional PDF text ──> Gateway (keys, per-page metering)
                                       │
                                       ▼
                        Page scheduler (batch lane | agent lane)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   Visual-token budgeter      Schema-prefix cache (APC)     Copy speculator
   (resize -> ~1k tokens;     (instruction+schema hit)      (OCR/PDF text -> draft
    pruning: re-attempt)                                      tokens; VLM verifies)
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                    vLLM core: Qwen3-VL-8B + LoRA, constrained JSON decode
                                       │
                                       ▼
                         markdown + JSON fields, confidence, tokens
```

## 6. Evidence

### Proven
- Visual-token budget lever (Qwen2.5-VL-7B, vLLM 0.24.0, A100-80GB, 200 DocVQA pages): 1536 px 0.935 acc / 11.4 img/s / 2,201 tokens; 1024 px 0.925 / 30.2 img/s / 1,017 tokens (2.7x, -1.0 pt); 768 px 0.885 / 47.2 (4.1x, -5 pts); 512 px 0.735; 384 px 0.445. Source: `/home/user/dexa/evals/RESULTS.md` 'Multimodal execution thesis'; 01-ledger `doc-vlm-frontier`. Caveat: `max_tokens=32`, so these are short-answer throughputs, not full-page parse.
- Accuracy parity vs GPT-4o on the same 200 pages: 0.925 vs 0.880 (relaxed match). Source: `/home/user/dexa/docs/FINDINGS.md`; 01-ledger `gpt4o-docvqa-head-to-head`.
- vLLM prefix caching, in-engine: cold vs warm TTFT ~280 -> ~24 ms at 4k, ~1,250 -> ~45 ms at 16k (Qwen2.5-7B text). Source: `/home/user/dexa/evals/RESULTS.md` 'Production reproduction on vLLM paged KV'. Engine-native, not ours.
- Decode-step reference for an 8B decoder: n=1 ~22 ms/step, flat across context (Llama-3.1-8B, vLLM 0.24.0, A100). Source: 01-ledger `vllm-decode-pathology`. Used only to argue a hundreds-of-token parse is decode-dominated single-stream; the VLM's split is unmeasured.
- Deployables: `/home/user/dexa/serve/modal_doc_vlm_serve.py` (Qwen2.5-VL-7B as 'doc-vlm', max_pixels 1048576); gateway with 29 passing tests (03-build-inventory.md).

### Bounded / contradicted
- "~40x cheaper than GPT-4o" decomposes into 3.4x fewer tokens (1,105 vs 322 for a 1280x800 screenshot, formula) times 12.5x from an assumed $0.20/M rate vs $2.50/M; the accuracy was measured at 1,017 tokens, not 322; the head-to-head script hardcodes the Qwen cost row ("$0.02-0.06/1k pages"); GPT-4o's measured $/page is unrecorded. Source: 04-conflicts.md conflict 10; 01-ledger caveats. FINDINGS.md's "matched-token-budget ~2.7x" misattributes the internal resize number.
- The 2.7x lever is naive resize "a client can apply before any API"; `evals/RESULTS.md` says it "sizes the prize but isn't yet the moat". The same file (Aug 2) concludes every inference-stack path is closed; `docs/FINDINGS.md` (Sep 2) picks a stateful provider; neither endorses this direction (03-build-inventory.md, disagreement 10).
- Content-aware pruning is OPEN, not failed: Qwen2.5-VL `get_rope_index` expects the full grid ([3,1671] computed vs [3,881] masked); no accuracy numbers. Source: 01-ledger `vlm-moat-mrope`.
- Cautionary analog: CUA screenshots are 86% redundant per patch, yet the capturable ceiling measured 2.34x at cos >= 0.98 (01-ledger `delta-tolerance-sweep`) — "redundancy" and "realized speedup" diverge.

### Unproven

| claim | experiment that proves or kills it | est. cost |
|---|---|---|
| Output is mostly a copy of input text | Parse 1,000 olmOCR-Bench/DocVQA pages with Qwen3-VL-8B; measure share of output tokens verbatim in the PDF text layer / a CPU OCR pass | ~2 GPU-h, 2 days |
| Copy-speculation gives >= 2x single-stream decode speedup | Same pages, n-gram prompt-lookup draft from OCR text vs none; acceptance length and s/page | ~5 GPU-h, 3 days |
| Prefill vs decode split for full-page parse | Time prefill and decode separately at 1024 px for 50/200/500-token outputs | ~2 GPU-h, 1 day |
| LoRA-SFT 8B/4B beats base and reaches >= 85.2 olmOCR-Bench | ~50k synthetic pairs at ~$0.12/page (~$6k); LoRA SFT on one H100/H200; eval on olmOCR-Bench, DocVQA, held-out extraction | ~40 GPU-h + ~$6k, 2 weeks |
| A trusted schema-extraction eval exists | Assemble from public KIE sets plus synthetic schemas; publish with harness | 0 GPU-h, 1 week |
| Content-aware pruning beats resize on Qwen3-VL | Re-implement masked-token positions on Qwen3-VL; sweep 0.5/0.25 vs resize | ~10 GPU-h, 2 weeks |
| Pages/GPU-second under concurrency | Load-harness sweep N = 8..256 on A100 and H100 | ~10 GPU-h, 3 days |
| Buyers value per-page latency in agent loops | Three design-partner pilots with logged s/page and accept/reject | 0 GPU-h, 4 weeks |

## 7. MVP and 6-week build plan

**Ships first:** a hosted `/v1/parse` endpoint (page image or PDF in; markdown + JSON fields out), priced per page, with a public benchmark page and rerunnable harness. Model v0 is stock Qwen3-VL-8B on the specialized engine; the fine-tune lands in week 4.

- **Week 1 — baselines and eval.** Port `evals/modal_vlm_frontier.py` and `evals/modal_incumbent_docvqa.py` to Qwen3-VL-8B and olmOCR-Bench; run baselines (1)-(5). Measure copy fraction and prefill/decode split. Deliverable: baseline table.
- **Week 2 — copy speculator.** Prompt-lookup draft from PDF text / CPU OCR inside vLLM; acceptance and s/page. Kill gate 1. Deliverable: speculation report.
- **Week 3 — engine v0.** Budgeter at the 1024 px point on Qwen3-VL; constrained JSON; schema-prefix caching; scheduler skeleton from `evals/modal_verifier_sched.py`. Wire into `dexa_platform/gateway` (drop the `stream=False` hard-set at `app.py:127`; per-page metering in `pricing.py`). Deliverable: end-to-end endpoint on Modal from `serve/modal_doc_vlm_serve.py`.
- **Week 4 — data and fine-tune.** ~50k synthetic pairs (olmOCR recipe); LoRA SFT 8B and 4B; eval. Kill gate 2. Deliverable: model v1 or a documented no-go.
- **Week 5 — throughput and pricing.** Adapt `voice-inference/vkv/loadgen` + `metrics` (text/image prompts, aiohttp); sweep concurrency on A100 and H100; derive $/1k pages from measured pages/s and GPU rate; publish the benchmark page.
- **Week 6 — pilots.** BYOC container (extend `dexa_platform/docker-compose.byoc.yml`, which already runs `vllm/vllm-openai:v0.24.0` with Qwen2.5-VL-7B); three design partners; third-party listing if available.

Reused: the two eval scripts, `serve/modal_doc_vlm_serve.py`, `dexa_platform/{gateway,control}`, `docker-compose.byoc.yml`, the `modal_verifier_sched.py` pattern, voice-inference loadgen/metrics. New: speculator, budgeter/pruner, grammar decoder, scheduler policy, synthetic-data pipeline, fine-tune, benchmark page. Not reused: KV connectors, tiering, cartridges, compaction (wrong workload shape).

## 8. Pricing model

Per page, two SKUs (parse; parse + extract with schema), volume tiers, batch discount, ZDR option — the units buyers already use. The architecture makes a per-page price expressible because cost per page is bounded by design: the budgeter fixes visual tokens near 1,017, the schema prefix is cached, and output is bounded by the page. A per-token provider cannot quote a page price because image tokens vary with resolution and tokenizer rules (Claude 28 px patches capped at 1,568/4,784; OpenAI 32 px patches x 1.2; Gemini 280/560/1120 per page).

Illustrative floor, derived and conditional: 30.2 pages/s at 1024 px (short answers) is 108,720 pages/GPU-hour; at Modal's A100-80GB rate of $2.50/hr (05-research-gpu-pricing.md) that is about $0.023 per 1,000 pages of field QA, consistent with the repo's printed "$0.02-0.06 / 1k pages". Full-page parse emits hundreds of tokens and will be materially slower — unmeasured. Against list prices of $10-$50 per 1,000 pages for cloud extraction and $10-$40 for Reducto parse/extract, the margin room is what the week-5 benchmark must confirm. Free tier: 10,000 pages, matching Unstructured's.

## 9. Competitive facts

| who | adjacent thing shipped | not shipped (per research files) | source |
|---|---|---|---|
| Morph | Fast Apply 10,500 tok/s (custom kernels + task speculator); Compact 33k tok/s; Reflexes; Glance; open-model hosting | No document parsing/OCR product listed | 05-research-morph.md |
| Reducto | r-1 parse at $0.01/page (2026-09-01); Extract $20/1K; throughput guarantees 200/350/500+ concurrent pages | Model size, latency, self-hosting undisclosed | 05-research-docai.md |
| Mistral | OCR 4.1 $4/1K ($2 batch); self-reported olmOCR-Bench 85.20; enterprise self-host | Per-page latency unpublished | 05-research-docai.md |
| AllenAI olmOCR-2 / rednote dots.mocr | Open 7B (82.4) and 3B MIT (83.9) OCR models; dots in vLLM 0.11.0; olmOCR data+code released | No hosted endpoint found | 05-research-docai.md |
| DeepSeek-OCR | 97% precision at <10x compression; 200k+ pages/day on one A100-40G | No hosted API found | 05-research-docai.md |
| Qwen3-VL | Apache-2.0 2B-235B; 8B DocVQA 96.1; vLLM/SGLang | No page-priced product | 05-research-docai.md |
| Azure / Google / AWS | Per-page APIs $1.50-$50 per 1K; commitment tiers | No open weights or BYOC | 05-research-docai.md |
| Extend / LlamaParse / Unstructured | Credits or flat $0.015/page; agentic modes to 45 credits/page | Underlying models undisclosed | 05-research-docai.md |
| Together / DeepInfra / Fireworks / Baseten | Hosted Qwen3-VL per token; speculative decoding in-engine (Together ATLAS; Baseten NGram/Eagle/MTP); Fireworks says >95% of its tokens are customer-specialized models | No per-page SKU, no page-tuned engine | 05-research-docai.md, 05-research-providers.md |
| vLLM / SGLang | Video-token pruning (EVS); prefix caching; grammar decoding | Image-token pruning (RFC #45098 open); SGLang EVS incompatible with Qwen2.5-VL positions | 05-research-docai.md |
| Wafer; Relace; Cognition | Optimization agents and dedicated endpoints; apply and search models at 10k+ / 2,800 tok/s | No document product | 05-research-wafer.md, 05-research-morph.md |

## 10. Risks and pre-registered kill gates

| risk | measurement | kills | proceeds |
|---|---|---|---|
| Output is not copy-like | Copy fraction on 1,000 pages (week 1) | < 40% of output tokens verbatim in source | >= 70% |
| Speculation does not pay | s/page with vs without copy draft (week 2) | < 1.5x | >= 2x |
| Parse is prefill-bound | Prefill/decode split at 200-500 output tokens | decode < 30% of page time and pruning blocked | decode >= 50% or pruning unblocked |
| Fine-tune does not beat base / specialists | olmOCR-Bench, DocVQA, held-out extraction (week 4) | < base or < 82.4 | >= 85.2 and >= base |
| Pruning stays blocked on Qwen3-VL | Matched-budget accuracy vs resize | no gain at 0.5 after 2 weeks: drop, rely on resize | >= +1 pt at 0.5 |
| Throughput claim fails under load | Sweep N = 8..256, p95 s/page | < 2x vs vanilla vLLM at equal accuracy | >= 3x |
| Unit economics below list | Measured pages/GPU-h x GPU rate vs $10/1K | COGS > $5/1K pages | COGS <= $1/1K |
| Buyers ignore page latency | Pilot logs, interactive vs batch share | 0 of 3 pilots use the agent lane | >= 2 of 3 |
| Base-model shelf life ("fast apply is dead" argument) | Frontier $/page and s/page tracked quarterly | frontier within 2x of our $/page | gap >= 5x |

## 11. Founder decisions

- **Market size.** Not judged. Informing evidence: the per-page price lists (Section 2); no volume disclosures for document APIs exist in the research files.
- **Whether incumbents' adjacency matters.** Reducto r-1 at $0.01/page, Mistral OCR 4, open olmOCR/dots.mocr, and Morph's ability to add a doc model are facts, not verdicts. Options: compete on latency + per-page price with a public benchmark, or pick another task from the table. Evidence: week-1 baselines.
- **Parse-first vs extract-first.** Parse has public benchmarks and a higher copy fraction; extract has the higher price points ($20-$50/1K). Evidence: copy fraction and accuracy per SKU.
- **Hosted vs BYOC vs open weights.** Gateway and compose file support hosted and BYOC today; Mistral self-hosts for enterprise; olmOCR/dots are open; Morph does not release weights. Evidence: pilot procurement asks.
- **Model choice.** Qwen3-VL-8B (96.1) vs 4B (95.3) vs 2B (93.3) vs dots.mocr 3B vs olmOCR-2-7B as base; from-scratch is out of 6-week scope. Evidence: week-4 accuracy/throughput per size.
- **GPU class.** Ledger data is A100/L40S. Rates: A100-80GB $1.39-$2.79/hr (RunPod, Modal, Lambda), H100 $3.29-$3.99, L40S $0.99-$1.95, MI355X rental $2.29-$2.95 (Wafer). Evidence: week-5 sweep on two classes.
- **Speculator form.** Prompt-lookup only vs trained draft head (Morph: 3.07x vs 1.93x for a task-trained draft over a generic one). Evidence: week-2 acceptance lengths.
- **Publish the harness and benchmark or not.** Morph's benchmark pages were unreadable (HTTP 429) in research; Wafer publishes openly. Evidence: pilot conversion.
- **Pricing unit.** Per page vs per credit vs per token; batch discount depth. Evidence: measured COGS per SKU.

## 12. Combinations

- **Document/CUA gateway direction (dexa_platform):** Fast Parse is the first model behind that gateway; `pricing.py`'s cross-provider image-token accounting becomes the buyer-facing savings meter.
- **Verifier-guided search direction:** for hard pages, best-of-N with a schema/consistency verifier and early stop (2.63x fewer tokens at equal pass@16 on code) is a cheap accuracy lane; unmeasured on documents.
- **Stateful-session direction:** multi-page documents share a prefix; vLLM prefix caching covers it, and the ledger shows KV loading loses to re-prefill below ~64k on the disk tier, so no connector is needed here.
- **Voice direction:** no technical overlap; the reusable asset is the load harness and the pre-registered-gate methodology.
- **Wafer-style optimization agent:** the visual-token path and grammar decoder are what such an agent would tune; the engine itself could be licensed to self-hosters.
- **Morph-catalog expansion:** if the playbook works once, compaction and tool routing reuse the speculator and scheduler; each needs its own data and eval, which the repos do not contain.