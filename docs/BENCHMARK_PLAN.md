# Independent benchmark plan — proving cross-instance KV reuse

The goal is to validate Dexa's serving claims with **established, independent
harnesses** (not a Dexa-built benchmark), so the numbers are credible to ML
engineers. This plan is grounded in a cited survey — see
`docs/research/2026-07-09-independent-benchmarks.md`.

**Prerequisite:** all of this drives a *working* vLLM server, so it cannot run until
the connector completion in `docs/CONNECTOR_COMPLETION.md` lands (a real request must
round-trip KV through vLLM first).

## Claims and how each is validated

| # | Claim | Validated by |
|---|---|---|
| 1 | Cross-instance KV reuse lowers TTFT / raises throughput under load | independent serving harness (below) — **serving metrics** |
| 2 | Matches or beats LMCache and vanilla vLLM prefix caching | same harness, connector config swapped |
| 3 | (only if compaction is headline) compacted KV preserves long-context accuracy | RULER / LongBench / ∞Bench — **not yet scoped, see gaps** |
| — | Resume is bit-identical (lossless) | **Dexa's own** two-process test — the harnesses below do *not* check correctness |

## The two harnesses (both independent, both drive a plain endpoint)

### 1. `vllm bench serve` (vLLM first-party CLI, canonical)

Reports TTFT, TPOT, ITL, E2EL, request/token throughput with P50/P99
(`--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99`). Realistic
load via `--max-concurrency`, `--request-rate`, `--burstiness` (=1 → Poisson),
`--num-warmups` (excluded from metrics). Endpoint-agnostic — drives a Dexa vLLM
server unchanged. Apache-2.0.

Purpose-built prefix/KV-reuse workloads:
- `--dataset-name prefix_repetition` (`--prefix-repetition-prefix-len`, `-suffix-len`,
  `-num-prefixes`, `-output-len`) — shared prefixes with varying suffixes.
- `benchmark_serving_multi_turn.py` (`benchmarks/multi_turn/`) — explicitly designed
  to stress KV offload/reuse across turns; `--max-active-conversations` forces
  eviction + retrieval from the offloading backend.
- `--dataset-name timed_trace --self-timed` — replays timestamped traces (Mooncake).

Baseline toggle: vanilla vLLM prefix caching is ON by default in V1; disable with
`--no-enable-prefix-caching`.

### 2. Mooncake production traces (the independent real workload)

Moonshot/Kimi anonymized production serving traces (FAST'25 release), JSONL with
`hash_ids` block-hashes (block size 512) encoding genuine prefix sharing:
`conversation_trace.jsonl` (~12k reqs), `toolagent_trace.jsonl` (~24k),
`synthetic_trace.jsonl` (~4k). Each record: `{timestamp, input_length,
output_length, hash_ids}`.

Replay options:
- **NVIDIA AIPerf** (successor to GenAI-Perf):
  `aiperf profile --model <m> --endpoint-type chat --streaming --url localhost:8000 \
   --input-file <trace>.jsonl --custom-dataset-type mooncake_trace --fixed-schedule`
- **vLLM native:** `vllm bench serve --dataset-name timed_trace --self-timed`

Mooncake also confirms Dexa's V1 `KVConnectorBase` is the industry-standard surface
(`MooncakeStoreConnector` is a shared cluster KV pool on the same seam) and is a
natural cross-instance baseline alongside LMCache.

## The first experiment

Replay the Mooncake `conversation_trace` (high prefix-sharing) against **three
identical server configs** — same model, GPU, trace, connector swapped via
`--kv-transfer-config` — and compare **P50/P99 TTFT and throughput**:

1. **Dexa** connector
2. **vanilla vLLM prefix caching** (`--no-enable-prefix-caching` off = baseline on)
3. **LMCache** connector

## Design gotchas that change the result (read before running)

1. **Affinity ≠ transfer — the critical one.** `prefix_repetition` and session-ID
   routing test prefix *reuse/affinity* (a session pinned to a backend hits its
   **local** cache). To prove Dexa's actual differentiator — KV *moving between
   instances* — the routing must force each request onto a **different/cold** instance
   than where its KV was computed (e.g. round-robin across ≥2 instances), so the only
   way to hit is the shared store. **Design routing to defeat local prefix caching**,
   or you will benchmark vLLM's built-in cache, not Dexa.
2. **Correctness is separate.** These harnesses measure latency/throughput only. The
   lossless claim is proven by Dexa's own two-process greedy-decode diff
   (`docs/CONNECTOR_COMPLETION.md`), not here.
3. **No turnkey Dexa-vs-LMCache suite exists** — assemble it from `vllm bench serve`
   runs with swapped connector configs.
4. **Pin a vLLM version.** CLI naming (`benchmark_serving.py` → `vllm bench serve`)
   and defaults (prefix caching now ON in V1) have shifted. Match the version pinned
   for the connector.
5. **Do not cite the Mooncake blog's "46× TTFT" figures** — refuted in verification.
   Generate your own numbers.

## Gaps (honest)

- **Compaction quality (claim 3)** — RULER / LongBench / ∞Bench are the right
  standard long-context benchmarks, but the survey could not verify run-details or
  KV-compaction relevance. Scope separately, and only if compaction stays in the
  headline (current honest read: the system-layer wedge, not compaction, is the pitch).
- **Cross-instance router** — an apples-to-apples cross-instance test needs a router
  (LMCache router, Mooncake Store, or NIXL/Dynamo) that actually transfers KV. Confirm
  Dexa's connector slots into one before claiming cross-instance parity.
