# Dexa — Findings & Learnings

A consolidated record of everything experimented with, measured, and built. The honest ledger
of raw numbers lives in [`evals/RESULTS.md`](../evals/RESULTS.md); this is the synthesis.

---

## The meta-finding (the through-line)

Nearly every inference-efficiency thesis, measured honestly on real A100s, **collapsed to
~2–3×, commoditization, or an architectural wall** — *except one*, which went the other way and
**grew with scale**. That consistency is itself the result: there is no free 10× hiding in
inference efficiency, and the one durable win is **architectural (statefulness), not a kernel
trick**.

---

## Efficiency theses — falsified or bounded

| Thesis | Measured result | Verdict |
|---|---|---|
| **Verifier-guided early-stop** (best-of-N, abort losers mid-decode) | **2.6×** fewer generated tokens at identical pass@16 (HumanEval); 93–96% of the gain held with a weak verifier; ~1.85× under load | Real but **buyer-replicable**, not a moat |
| **Decode pathology** | **174×** blowup for n=8 branched decode at 128k in vLLM… but **SGLang handled it at 2.2×** | **Not a moat** — tree-native space occupied |
| **KV interchange (formats)** | fp8/int8 round-trip **lossless**; int4 breaks | Narrow |
| **KV interchange (weights)** | base↔instruct KV injection **diverges in 2–5 tokens** despite 0.98 cosine; learned bridge failed (0%→6%) | **Weights don't interchange** |
| **LoRA over shared base** | KV reusable only to **~3%** weight change | Too narrow to be a product |
| **Agentic serving** | naive win reduces mostly to **prefix caching** | **Table stakes** |

---

## Multimodal / document VLM — the first surviving prize

- **DocVQA head-to-head** (same 200 pages, measured): open Qwen2.5-VL-7B **0.925 vs GPT-4o
  0.880**, at **~40× lower cost/page** (fewer visual tokens + cheaper model). A separate
  matched-token-budget framing showed ~2.7×.
- **Content-aware visual-token pruning** to widen the gap: **blocked by Qwen2.5-VL's mRoPE grid
  coupling** — a real architectural wall, recorded not fixed.
- Deployed a working **OpenAI-compatible VLM endpoint** on Modal to prove the drop-in.

**Read:** a real *economics/config* win, not a novel-kernel moat. Copyable.

---

## Computer-use agent perception — a real prize, capped

- Agent screens are **~86% redundant** frame-to-frame (7.3× headroom).
- But the vision encoder **spills a small change across ~3.4× the tokens** → naive reuse
  captures **~2×, not 7×**.
- Tolerance sweep: the spill is **hard, not soft** (median moved-token cosine **0.83**) —
  loosening tolerance doesn't rescue it. **5× path closed; ceiling ~2.3×.**

---

## The one that survived and grew: stateful warm sessions

**Question:** when a long-idle agent session resumes, is *restoring* its KV cache cheaper than
*re-prefilling* it? Reproduced **three independent ways, all agreeing**:

| Reproduction | 4k | 16k | proves |
|---|---:|---:|---|
| Raw physics (HF tensor copies) | 11.8× | 11.5× | the move-bytes-vs-recompute physics |
| Production vLLM (in-GPU prefix cache) | ~12× | ~25–34× | reuse beats recompute in the real engine |
| **Off-GPU offload→restore (vLLM+LMCache)** | **10.0×** | **17.1×** | **KV evicted to CPU, restored on resume — the hard case** |

- **Widens with context** (restore is bandwidth-bound/linear; prefill is compute-bound/
  super-linear) — the opposite of every other thesis, which shrank toward parity.
- Correctness: restore-then-decode advances the KV correctly (byte-identical in faithful mode).

### Residency economics (cost model)
- **Latency win is unconditional:** resume a 64k session in ~0.3s vs ~8.6s = **28.8×**.
- **$ win is 2–6×** on long-context/idle-gapped sessions — *but only with tiering*.
- **Break-even idle per tier:** warm GPU HBM **~2 min**, RAM **~11 min**, NVMe **up to ~5 hr**
  (at 64k). Hoarding GPU memory *loses money*; the $ win comes from demoting idle KV to cheap
  NVMe. Bigger models win **more**; fp8 KV ~doubles every break-even.

### Honest scope + the delta over LMCache
- **Narrow use-case:** long-lived agents whose idle gaps exceed prompt-cache TTLs
  (human-in-the-loop, async/scheduled). Within TTL, stateless ≈ stateful.
- **Delta over LMCache is *not* the restore mechanism** (LMCache does that). It's the
  **economic tiering policy** — retain-vs-re-prefill and GPU/RAM/NVMe placement from measured
  break-evens — the decision LMCache's LRU never makes.

---

## What got built

**`dexa_platform/`** (Python · 29 tests · live GPU backend)
- OpenAI-compatible **gateway** with per-request cost/savings telemetry + a session redundancy
  meter + exact-frame cache.
- **Control plane:** accounts, SHA-256-hashed API keys, PLG signup + free credits, durable
  credit-metered usage, KV-cached key resolver (hot path never blocks on Postgres). Multi-tenant
  hosted + **BYOC**.
- **Stateful session service:** `create_session` / warm turns, tiering engine, savings
  telemetry. **Demoed live** — 12k-token session cold-prefills ~5.4s, restores warm ~1.5s
  (3.7× end-to-end).

**`serve/`** — Modal **vLLM + LMCache** backends, deployed and warmed.

**`edge/`** — Cloudflare scaffold: **a session is a Durable Object** (durable state +
single-writer turns + backend affinity), Workers + KV + Postgres/Hyperdrive + Queues + R2.
**Typechecks clean**; deploy runbook + seed/smoke scripts. Not yet run in CF.

**`docs/`** — the [stateful-sessions thesis](STATEFUL_SESSIONS.md) + an architecture artifact.

---

## The direction this points to

The defensible company isn't a cheaper-VLM reseller or a new kernel — it's a **session-stateful
inference provider for long-idle / long-context agents**, where:
- the **moat** is the economic tiering decision + session orchestration + workload fit,
- the **DX** is one optional `session` field on the OpenAI call you already make,
- the **stack** is Cloudflare (control plane + session Durable Objects) over a Modal GPU tier.

### Open / unproven (the honest edges)
1. Tiering policy is **measured & designed but not yet *enforcing* eviction** in production
   (LMCache still runs its own LRU underneath).
2. In-engine runs **capped at ~16k** (vLLM V1 EngineCore OOM under long context); HF physics
   covers 32k–64k.
3. The **segment dependency-graph / RoPE-relocation reuse is design, not shipped code**.
4. The **Cloudflare edge needs its first real `wrangler deploy`** (needs a CF account + Postgres).

---

*All work lives on `claude/dexa-scaling-vllm-phydxe` and `main`. `evals/RESULTS.md` is the
raw-number ledger; each finding maps to a `modal_*.py` experiment.*
