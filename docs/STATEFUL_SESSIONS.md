# Stateful inference for long-lived agents

A thesis doc for an inference provider whose primitive is a **durable, warm session**
rather than a stateless request. It is grounded in one measured result — see
`evals/RESULTS.md` (*"Stateful warm-session vs re-prefill"*) and
`evals/modal_stateful_session.py` for the experiment.

---

## TL;DR

Every inference provider on the market is **stateless**: each request is independent, the KV
cache is best-effort and evicted within minutes, and a long-lived agent re-pays to prefill its
entire context every time it resumes. That is a workload/infrastructure mismatch — agent
sessions are *stateful by nature* (a growing context of conversation, tool history, a corpus,
accumulated screenshots).

Build the provider that matches the workload: a session is a first-class object whose KV state
is kept **warm** (or fast-restorable), so resuming a session **restores** its state instead of
**recomputing** it.

We measured the core physics on an A100. Restoring a saved KV cache beats re-prefilling it by
**11.8× at 4k context, growing to 28.9× at 64k** — and the gap *widens* with context. Projected
onto a 64k-context agent resumed 50 times across idle gaps: **18.3× less GPU prefill time.**

This is the first thesis in the project whose advantage **grows with scale** instead of
collapsing toward parity, because it rests on physics that don't wash out (below).

---

## The problem: stateful workloads on stateless infrastructure

A real agent session is long-lived and accretes context:

- a system prompt + a large tool schema (static, resent every step),
- a growing conversation / tool-call history,
- often a big attached corpus (codebase, enterprise docs) or a stream of screenshots.

And it **idles** — waiting on a human approval, a slow tool, another agent, or a scheduled
wake. These idle gaps routinely exceed a stateless provider's prompt-cache TTL (Anthropic's is
~5 minutes). On resume, the cache has been evicted, so the provider **re-prefills the entire
context from scratch** — the dominant cost and latency of the turn.

Stateless providers mitigate this with implicit prompt caching (OpenAI, Anthropic, DeepSeek,
Fireworks) and prefix caching (vLLM APC, SGLang RadixAttention). But those are *ephemeral and
evictable*: short TTLs, evicted under memory pressure, and bounded by what a shared cache holds.
For a long-lived agent with real idle gaps, they miss exactly when it matters.

---

## The measured result

`modal run evals/modal_stateful_session.py` — Qwen2.5-7B, bf16, A100-80GB. For each context
size we measured the cost of the two things a session can do on resume: **re-prefill** the
context (what a stateless provider pays after eviction) vs **restore** the saved KV from a
warm/pinned CPU tier or an NVMe tier (what a stateful provider pays).

| context | KV size | re-prefill (stateless) | restore CPU (stateful) | **speedup** | restore NVMe |
|--------:|--------:|-----------------------:|-----------------------:|------------:|-------------:|
| 4k  | 0.23 GB | 296 ms   | 25 ms  | **11.8×** | 1.2× |
| 16k | 0.94 GB | 1,321 ms | 115 ms | **11.5×** | 3.4× |
| 32k | 1.88 GB | 3,167 ms | 182 ms | **17.4×** | 4.9× |
| 64k | 3.76 GB | 8,567 ms | 297 ms | **28.9×** | 7.0× |

Projected onto a long-lived session (context re-warmed after each idle gap that evicts a
stateless cache):

| context | turns | stateless re-prefill time | stateful restore time | **multiple** |
|--------:|------:|--------------------------:|----------------------:|-------------:|
| 32k | 50 | 158.4 s | 12.3 s | **12.9×** |
| 64k | 50 | 428.4 s | 23.4 s | **18.3×** |

Decode/step is ~equal in both regimes (39–108 ms) — it is *not* the differentiator; prefill is.
Correctness was checked: after a CPU round-trip the restored KV still advances one token per
decode step. 128k OOM'd — an HF non-paged artifact (restore briefly doubles memory), not a
fundamental limit; paged KV handles it.

---

## Why the win grows with scale (the physics)

- **Re-prefill is compute-bound and super-linear.** Rebuilding the KV means a full forward pass;
  attention cost grows faster than linearly with context length.
- **Restore is bandwidth-bound and linear.** Moving KV bytes from CPU/NVMe to GPU scales linearly
  with the number of bytes.

So the two diverge as context grows: 11.8× at 4k → 28.9× at 64k, and it keeps widening. This is
the opposite of every efficiency thesis in `RESULTS.md` (verifier early-stop, decode pathology,
KV interchange, delta-perception), all of which shrank toward ~2× or commoditization. The move-
bytes-vs-recompute-bytes ratio is architecture-independent — the absolutes shift on production
infra, the ratio does not.

---

## The architecture

### 1. Session as the primitive
`create_session(model, context) -> session_id`. The context (system + tools + corpus +
conversation) is prefilled **once**; its KV is materialized and owned by the session, not the
request. Subsequent calls reference the `session_id` and pay only for **new** tokens.

### 2. A session-aware memory hierarchy
- **Warm (GPU HBM):** active sessions keep KV resident — next turn is immediate.
- **Offloaded (CPU / NVMe):** idle sessions are evicted from HBM to cheaper memory, freeing the
  GPU for other sessions.
- **Restore on wake:** resuming streams the KV back (25–300 ms warm, sub-1.3 s from NVMe in this
  naive measurement) instead of a multi-second full re-prefill.

### 3. Session-aware scheduling / routing
Route a session's request to where its KV lives; manage the tiering and eviction by session
value/recency. This is a router whose unit is a *session*, not a stateless request.

### 4. Pricing that matches the architecture
**Session-residency** (renting warm/offloaded state) **+ per-new-token** (compute), instead of
per-total-token-including-reprefill. A per-token stateless provider cannot express this price —
the pricing model itself is the architectural tell.

---

## Where it wins — and where it doesn't

**Wins (target these):** long-lived / large-context / **idle-gapped** agents — human-in-the-loop
approvals, async and scheduled agents, multi-agent handoffs, enterprise-corpus sessions,
multi-turn coding agents over a big repo. The advantage is the **idle-gap / eviction tail** and
it grows with context size.

**Wash (don't target):** short, bursty, low-context chat with no idle gaps. *Within* a stateless
cache's TTL there is no eviction, so stateless ≈ stateful. This is a narrow-and-deep workload
play, not "cheaper inference for everyone" — which is the point: narrow-and-defensible beats
broad-and-commoditized.

**Tier nuance:** the warm/pinned-CPU tier is the prize (11–29×, growing). NVMe is a secondary
tier (1.2–7×, only pays at ≥16k because naive `torch.load` deserialization dominates). A
production zero-copy / direct-IO store beats these naive copies — the measured numbers are
conservative floors, not ceilings.

---

## Product / PLG surface (low-effort, provable value)

The primitive stays an OpenAI-shaped API plus a `session_id`. A developer creates a session with
their context once, then sends turns against it. The value is **felt on turn #2**: the cold first
call prefills; every resume after is near-instant. That is provable-value-with-minimal-effort —
no traces to upload, no eval to configure — and it is a benefit the stateless incumbent
*structurally cannot show them*.

---

## Prior art and the open gap

Be precise about novelty. Persistent/offloaded KV is **not** new as a technique — prompt caching
(OpenAI/Anthropic/DeepSeek/Fireworks), prefix caching (vLLM/SGLang), and KV-offload systems
(AttentionStore, MoonCake, LMCache) all exist. Server-side conversation *text* state exists too
(OpenAI Responses / Assistants threads) — but that stores messages and re-prefills, it does not
keep the *compute state* warm.

The open gap is the **product abstraction**: a durable, self-serve, warm **session** for agent
builders — with idle-persistence beyond cache TTLs, residency guarantees (no eviction surprises),
and contexts larger than a shared cache holds — priced as state + delta. Stateless per-token
providers don't offer it because it fights their multiplexed, throughput-optimized core. Claim
the *session-as-a-product* architecture, not the KV trick.

---

## Proven vs. not proven

**Proven (measured), three independent ways that agree:**
1. **Raw physics (HF):** restore vs re-prefill 11.8×→28.9×, growing with context; restore-then-
   decode correctness checked. (`modal_stateful_session.py`)
2. **In-GPU prefix cache (vLLM):** cold vs warm TTFT ~12×→~29× in the real engine.
   (`modal_vllm_warmstart.py`)
3. **Off-GPU offload→restore (vLLM + LMCache), the hard case:** KV evicted to CPU then restored
   with vLLM's prefix cache OFF — 10.0× at 4k, 17.1× at 16k, LMCache logs confirming a CPU
   retrieve at ~22 GB/s. (`modal_lmcache_restore.py`)

**Not yet proven (next):**
1. **Residency unit economics** — is renting cheap CPU/NVMe for idle KV + a small restore
   genuinely cheaper than re-prefill at realistic reuse frequencies and session lifetimes? The
   compute side is proven; the memory-cost model is not.
2. **Scale past ~16k in-engine** — both vLLM runs stopped at 16k (V1 EngineCore OOM/kernel death
   under long context in this harness); HF raw-physics covers 32k–64k where the trend continues.
3. **Idle-persistence across a real gap + NVMe tier** end-to-end (LMCache disk backend), and
   concurrent multi-session residency/eviction under load.

---

## Build plan

1. **vLLM + LMCache warm-session demo** — a live session that offloads on idle and restores on
   wake through a production KV store. Deliverable: "resume a 64k-context agent in ~300 ms
   instead of ~9 s," on real infra.
2. **Residency cost model** — price session-hours (HBM/CPU/NVMe tiers) vs re-prefill; find the
   reuse frequency where stateful is unit-economically cheaper.
3. **Session API + tiered scheduler** — `create_session` / referencing turns; warm→offload→
   restore lifecycle; session-aware routing.
4. **PLG surface** — OpenAI-compatible session endpoint; the turn-#2 warm-start is the demo.

---

## Reproduce

```bash
modal run evals/modal_stateful_session.py
```

Measures, per context size on one A100-80GB: `prefill` (re-prefill), `offload` (GPU→CPU),
`restore(CPU)`, `restore(NVMe)`, `decode/step`, plus a projected T-turn idle-gapped session.
