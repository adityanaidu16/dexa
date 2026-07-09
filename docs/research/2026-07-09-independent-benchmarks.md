# Independent benchmarks for KV-cache persistence & cross-instance reuse (cited survey)

*Generated 2026-07-09 by a multi-agent deep-research pass (5 search angles, 23 sources
fetched, 111 claims extracted, 25 adversarially verified — 23 confirmed, 2 refuted).
Feeds `docs/BENCHMARK_PLAN.md`.*

**Question.** What are the best *independent, established* benchmarks/harnesses to
validate Dexa's serving-level cross-instance KV-reuse claims (TTFT/throughput under
load; competitive vs LMCache and vanilla vLLM prefix caching; secondarily, compaction
fidelity), that the user can run without building a benchmark?

## Bottom line

The strongest ready-made options are **(1) vLLM's own first-party serving benchmark**
(`vllm bench serve` / `benchmark_serving.py`, plus `benchmark_serving_multi_turn.py`
and the Rust `vllm-bench`) and **(2) real Mooncake production traces** replayed via
**NVIDIA AIPerf** or `vllm bench serve --dataset-name timed_trace`. All are
endpoint-agnostic and independent of Dexa: Dexa exposes a normal vLLM server, the
harness drives it. First experiment: replay the Mooncake conversation/toolagent trace
(high prefix-sharing) against Dexa vs vanilla prefix caching vs LMCache, comparing
P50/P99 TTFT and throughput.

## Confirmed findings

1. **`vllm bench serve` measures every metric the serving claim needs** — TTFT, TPOT,
   ITL, E2EL, request/token throughput, with `--percentile-metrics` /
   `--metric-percentiles` (default 99). Canonical, actively maintained, endpoint-agnostic.
   *(high; 3-0)* — vllm benchmarks/README.md; docs.vllm.ai/en/stable/cli/bench/serve/;
   docs.vllm.ai/en/latest/benchmarking/cli/

2. **Realistic concurrent-load simulation** — `--max-concurrency` (concurrency ceiling),
   `--request-rate` + `--burstiness` (=1 → Poisson, gamma otherwise), `--num-warmups`
   (excluded from metrics). *(high; 3-0)* — docs.vllm.ai/.../bench/serve/; vllm PR #18475

3. **Purpose-built prefix/KV-reuse workloads** — `prefix_repetition` dataset
   (`--prefix-repetition-{prefix,suffix}-len`, `-num-prefixes`, `-output-len`) and a
   standalone `benchmark_prefix_caching.py` (fixed or ShareGPT prompts). Vanilla prefix
   caching toggled via `--enable-prefix-caching` (ON by default in V1). *Caveat: these
   exercise a prefix-sharing workload but do not by themselves prove cross-INSTANCE KV
   transfer.* *(high; 3-0)* — docs.vllm.ai/.../benchmarking/cli/;
   github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_prefix_caching.py; PR #10724

4. **Official multi-turn serving benchmark** — `benchmark_serving_multi_turn.py`
   (`benchmarks/multi_turn/`, RFC #20265 / merged PR #20267) is explicitly designed to
   stress KV offload/reuse: round-robins concurrent conversations
   (`--max-active-conversations`) to force eviction + retrieval from an offloading
   backend; reports TTFT/TPOT/E2E/RPS. *(high; 3-0)* — vllm issues/20265; PR #20267

5. **`vllm-bench`** (official Rust drop-in for `vllm bench serve`) has a multi-turn mode
   sending `X-Session-ID: {conversation_id}` per turn to drive KV reuse / router
   affinity; same metrics + output schema. *Caveat: header enables session-AFFINITY
   routing; true cross-instance transfer needs the router/connector to consume it.*
   *(high; 3-0)* — github.com/vllm-project/vllm-bench

6. **NVIDIA AIPerf replays real Mooncake traces** in one command
   (`--custom-dataset-type mooncake_trace --fixed-schedule`); equivalently
   `vllm bench serve --dataset-name timed_trace --self-timed`. Endpoint-agnostic,
   independent. *Caveat: measures request-level latency/throughput, not KV-internal
   reuse or bit-identical correctness.* *(high; 3-0)* —
   docs.nvidia.com/aiperf/benchmark-modes/trace-replay-with-mooncake-traces;
   docs.vllm.ai/en/latest/cli/bench/serve/; github.com/ai-dynamo/aiperf

7. **Mooncake publicly releases anonymized production traces** (FAST'25) as JSONL —
   `conversation_trace.jsonl` (12,031 reqs), `toolagent_trace.jsonl` (23,608),
   `synthetic_trace.jsonl` (3,993). Each record: `{timestamp, input_length,
   output_length, hash_ids}`; identical `hash_ids` (block size 512) mark shared token
   blocks + all preceding tokens — the exact structure to exercise KV reuse.
   *(high; 3-0)* — github.com/kvcache-ai/Mooncake; NVIDIA AIPerf docs

8. **Mooncake is the established cross-instance baseline and confirms Dexa's V1
   `KVConnectorBase` is the industry-standard surface** — `MooncakeStoreConnector`
   turns per-instance caches into a shared cluster KV pool (hash-based prefix caching,
   cross-instance reuse); `MooncakeConnector` does P2P prefill→decode transfer over
   RDMA; both plug the same `KVConnector` seam, chained via `MultiConnector` — the
   surface Dexa hooks. *(high; 3-0)* — github.com/kvcache-ai/Mooncake;
   vllm.ai/blog/2026-05-06-mooncake-store; docs.vllm.ai/.../mooncake_connector_usage/;
   vllm PR #10884

## Refuted (do not cite)

- "vLLM officially ships specialized benchmarks including prefix caching and long
  document QA" — *0-3.* (Overstated; the shipped items are load-generators + the
  prefix workloads above, not a long-doc-QA suite.)
- "Mooncake Store lowered P50 TTFT 46×, throughput 3.8×, E2E 8.6× on GB200 agentic
  traces" — *0-3.* Do not cite specific Mooncake speedup numbers as independent proof.

## Caveats

1. **Compaction quality (claim 3) uncovered** — no RULER/LongBench/∞Bench claim
   survived verification; those are real standard long-context benchmarks but their
   run-details and KV-compaction relevance need separate research before relying on them.
2. **No turnkey competitive suite** — the confirmed harnesses are the plumbing; stand
   up Dexa and LMCache as separate connector configs and drive them identically.
3. **Serving ≠ correctness** — every confirmed harness measures latency/throughput,
   none validates bit-identical KV; use a separate greedy-decode output-diff.
4. **Affinity ≠ transfer** — `X-Session-ID` and `prefix_repetition` drive reuse/affinity;
   proving cross-instance migration needs a router/connector that transfers KV.
5. **Time-sensitivity** — vLLM benchmarking moves fast (CLI renamed
   `benchmark_serving.py` → `vllm bench serve`; prefix caching now ON by default in V1).
   Pin a version.
6. **Source strength** — `vllm-bench` "standard/drop-in" rests on its own README;
   canonical harness remains `vllm bench serve`.

## Open questions

- How do RULER / LongBench / ∞Bench run against a served vLLM endpoint, and which
  subtasks are most sensitive to KV compaction? (priority-3, uncovered here)
- Is there any pre-packaged head-to-head KV-connector comparison (LMCache repo, vLLM
  production-stack, LMBench/kvcache-bench), or must it be assembled manually?
- Standard method to prove bit-identical zero-re-prefill correctness across instances —
  established harness, or ad-hoc greedy-decode diffing?
- Does a vLLM-ecosystem router (LMCache router, Mooncake Store, NIXL/Dynamo) consume the
  `X-Session-ID` signal to route/transfer KV, and can Dexa slot into it for an
  apples-to-apples cross-instance test?

## Key sources

- vLLM serving benchmark: docs.vllm.ai/en/stable/cli/bench/serve/ ·
  github.com/vllm-project/vllm/blob/main/benchmarks/README.md ·
  benchmark_prefix_caching.py · vllm-project/vllm-bench · issues/20265, PR #20267, #18475
- Mooncake: github.com/kvcache-ai/Mooncake · arxiv.org/html/2407.00079v1 (FAST'25) ·
  vllm.ai/blog/2026-05-06-mooncake-store · docs.vllm.ai/.../mooncake_connector_usage/ · PR #10884
- Trace replay: docs.nvidia.com/aiperf/benchmark-modes/trace-replay-with-mooncake-traces ·
  github.com/ai-dynamo/aiperf
- LMCache: blog.lmcache.ai/en/2025/04/29/bringing-state-of-the-art-pd-speed-to-vllm-v1-with-lmcache/ ·
  blog.lmcache.ai/2025-04-29-pdbench/
- Connector wiring: docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/base/ ·
  .../v1/example_connector/ · vllm PR #15960
- Long-context (unverified here): github.com/NVIDIA/RULER · github.com/THUDM/LongBench ·
  github.com/OpenBMB/InfiniteBench
