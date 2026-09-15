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

## Smoke run (run 6, 3 django tasks per arm, concurrency 1)

Both arms completed and graded: the server was healthy 274 s after a cold start with cached weights, tasks took 74 to
198 s, and the official grader resolved 2 of 3 vanilla tasks and 1 of 3 harness tasks. Model time was 55 to 82 s per
task against 15 s of tool time, so tool time was 13 to 17 percent of the wall clock at concurrency 1. Speculation
launched 12 times and hit once. Every Rule B miss was a matcher defect, not a wrong prediction: the predicted command
carried an absolute path and the model ran the relative one. That is fixed (paths under /testbed compare equal to
their relative form; deletions no longer count as modifications), so the smoke hit rate is not the number to read.
The throughput ratio from three tasks per arm (1.24) is noise: the harness arm saved 0.2 s per task and the gap is
model-time variance. Four of six tasks hit the 40-step cap without submitting.

## Cost

One H100 on Modal is about $4 per hour; sandboxes are about $0.20 per CPU-hour. The server is up only while a
workflow run is active. A smoke run is roughly 30 minutes of GPU time including the first model download. The full
harness phase (2 arms, 50 tasks, concurrency 8) is a few GPU-hours if tasks average around 10 minutes.

## What this does not measure

The engine-side half of the thesis (KV residency across tool waits, pre-filled tool results, speculative next step)
needs a modified vLLM and is the next phase. This directory measures only the harness-side rules under a real model
and real container latencies, which the replay could not do.
