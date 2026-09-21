# Phase 1 on Modal: live agent, vanilla vs speculative harness

This directory turns the replay result (`../../../docs/experiments/speculative-test-execution.md`) into a live
measurement: an open coder model runs real SWE-bench Verified tasks end to end, once with plain tool execution
and once with the two speculation rules, and the metric is **tasks per GPU-hour at equal resolve rate**.

## Pre-registered design

| item | value |
|---|---|
| tasks | `tasks_verified_50.json`: 50 SWE-bench Verified instances, fixed order: django 20, sympy 10, xarray 5, pytest 5, sphinx 4, requests 3, pylint 3; 25 rated under 15 minutes and 25 rated 15 to 60 minutes |
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on vLLM, tool calling via the `qwen3_coder` parser, one H100 |
| agent | bash + str_replace_editor + submit, 40 steps max, temperature 0.2, 300 s per command |
| arm `off` | every tool call runs when the model asks for it |
| arm `harness` | Rule A: after a modification of an existing file, pre-run the last test-like command. Rule B: after creating a `.py` file, pre-run it. A matching next call is served from the speculative run, anything else kills it |
| grading | after both arms finish: official SWE-bench eval script and log parser (swebench 4.1.0) in a fresh sandbox per task, model patch applied first, so grading never shares the clock with the timed loop |
| throughput | `n_tasks * 3600 / span` where span is first task start to last task end for the arm (GPU dedicated to that arm); also a steady-state figure `n * 3600 / (sum(agent_s) / concurrency)` that removes the tail where the last tasks run alone |
| gate | continue if harness/off >= 1.30 at equal resolve rate; kill if < 1.15; between: report and decide |
| order | smoke (3 tasks, concurrency 1, both arms) then harness (50 tasks, concurrency 8, both arms); engine arms come after and are not in this directory |

Arms run one after the other on the same server so they never share the GPU. Concurrency is the number of tasks in
flight per arm; it should be the same in both arms.

## Files

- `app.py`: the Modal app. One `VLLM` class (GPU container) runs `vllm serve` behind a `web_server` endpoint. The image is a CUDA base with vLLM installed from PyPI, the way Modal's own vLLM example builds it (the `vllm/vllm-openai` image cannot back a Modal Function because Modal cannot detect its Python). Weights are cached in the `spec-exec-hf-cache` volume; configuration travels as a Modal secret built from the deployer's environment.
- `live_agent_modal.py`: the harness. Creates one Modal Sandbox per task from the official image `swebench/sweb.eval.x86_64.<owner>_1776_<repo>-<n>:latest`, drives the model through the OpenAI-compatible endpoint, applies the two rules when `--spec harness`, grades, and appends one JSON record per task. Resumable: tasks already in the output file are skipped.
- `summarize.py`: per-arm table and the throughput ratio.
- `tasks_verified_50.json`: the task list with everything the grader needs.
- `runs/<run_id>/`: `config.json`, `<arm>.jsonl`, `<arm>.log`, `summary.md`, `summary.json`, committed by the workflow.

## How it runs

The Claude sandbox that wrote this cannot open gRPC connections, and Modal's client needs one. The GitHub Actions
workflow `.github/workflows/spec-exec.yml` is therefore the machine that talks to Modal.

1. Add repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Settings, Secrets and variables, Actions).
2. Dispatch the workflow on this branch with `phase=smoke`. It deploys the server, runs 3 tasks per arm, prints the summary in the job log, uploads the run directory as a workflow artifact, commits it under `runs/`, and stops the server.
3. Dispatch with `phase=harness`, `count=50`, `concurrency=8`.
4. `phase=stop` stops the server if a run was cancelled; `phase=deploy-only` leaves it up for manual use (it scales to zero after 15 idle minutes).

Locally, with the Modal CLI configured:

```bash
pip install "modal>=1.0" "openai>=1.50" "swebench==4.1.0"
export SPEC_VLLM_API_KEY=$(openssl rand -hex 16)
modal deploy app.py                      # prints https://<workspace>--spec-exec-vllm-vllm-serve.modal.run
python live_agent_modal.py --base-url https://... --api-key $SPEC_VLLM_API_KEY --spec off     --count 3 --out runs/local/off.jsonl
python live_agent_modal.py --base-url https://... --api-key $SPEC_VLLM_API_KEY --spec harness --count 3 --out runs/local/harness.jsonl
python summarize.py runs/local
modal app stop spec-exec-vllm
```

## Results so far

- Smoke (run 6, 3 django tasks per arm, concurrency 1): pipeline verified end to end; exposed a matcher defect (absolute against relative paths) that was fixed before the full runs.
- Run 7 (50 tasks per arm, arms in sequence): ratio 0.69 by span, 0.68 paired and boot-corrected, confounded by an outage at the arm switch (nine errored harness records), the first arm's image pulls, and 14 stalls. Kept as the first attempt; see `runs/20260915T044016Z-harness-7/analysis.md`.
- **Run 9 (50 tasks, both arms per task in the same worker, alternating order): gate result kill, arms at parity.** Throughput ratio harness/off 0.87 raw and 0.99 with stall time removed, typical task 3 percent slower with speculation on; resolve 27 against 24; rules hit 76 percent (rule B 92, rule A 32) and saved 2.1 s per task against a median loop of 138 s. vLLM's counters: prefix-cache hit rate 97 percent, zero preemptions, prefill 155 s against decode 8,599 s, no server-side call over 60 s while five client-side calls stalled 311 to 910 s (transport, not inference). See `runs/20260920T012018Z-harness-9/analysis.md` and `metrics_summary.md`, and the write-up in `docs/experiments/speculative-test-execution.md`.

- **Run 10 (concurrency sweep, vanilla arm, 8/16/32/64 sessions in flight, 350 task runs): the knee is between 32 and 64 sessions on one H100.** Tasks per GPU-hour by level span 174, 231, 295, 218; per-call median 1.1, 1.6, 2.1, 13.6 s; prefix-cache hit rate 96.9, 96.8, 96.5, 31.4 percent; preemptions 0, 0, 0, 1,517; time to first token 54, 63, 74 ms then 6.0 s. Contexts were the same at every level (11 thousand prompt tokens per call), so it is sessions times context against the 452-thousand-token KV budget. See `runs/20260921T003645Z-sweep-10/sweep.md`.

Next run worth doing: the 64-session level with vLLM's CPU-offloading KV connector enabled (`phase=sweep`, `levels=64`, with the offload config in `app.py`), to see whether a host-memory tier brings per-call latency and throughput back toward the 32-level numbers. That is the direct test of the residency lever.

## Cost

One H100 on Modal is about $4 per hour; sandboxes are about $0.20 per CPU-hour. The server is up only while a
workflow run is active. A smoke run is roughly 30 minutes of GPU time including the first model download. The full
harness phase (2 arms, 50 tasks, concurrency 8) is a few GPU-hours if tasks average around 10 minutes.

## What this does not measure

The engine-side half of the thesis (KV residency across tool waits, pre-filled tool results, speculative next step)
needs a modified vLLM and is the next phase. This directory measures only the harness-side rules under a real model
and real container latencies, which the replay could not do.
