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

- Smoke (run 6, 3 django tasks per arm, concurrency 1): pipeline verified end to end; exposed a matcher defect (absolute against relative paths) that was fixed before the full run.
- Phase 1 (run 7, 50 tasks per arm, concurrency 8, arms in sequence): **gate result kill.** Harness/off tasks per GPU-hour 0.69 by arm span, 0.89 steady state, 0.68 on the 41 paired tasks with sandbox boot removed, 0.77 with the three slowest tasks per arm dropped. Rules hit 76 percent (rule B 97, rule A 30) and saved 2.3 s per task against a ceiling of 10.9 s and a median loop of 115 s. Resolve rate 24 against 22 on the paired tasks. Nine harness records errored at the arm switch (a gateway non-completion body, now retried), the off arm paid 41 s per task of image pulls, and 14 model calls stalled for over 120 s; the full reading, with the per-task table, is in `runs/20260915T044016Z-harness-7/analysis.md` and in `docs/experiments/speculative-test-execution.md`.

Changes since run 7, for any rerun: `--spec both` runs every task in both arms inside the same worker with alternating order (removes arm-order and image-cache confounds; the workflow's default `arms=both`), the whole server log is streamed for the run, vLLM `/metrics` is snapshotted per arm (prefix-cache hits and queries, preemptions, prefill and decode time), and a non-completion body from the gateway is retried.

## Cost

One H100 on Modal is about $4 per hour; sandboxes are about $0.20 per CPU-hour. The server is up only while a
workflow run is active. A smoke run is roughly 30 minutes of GPU time including the first model download. The full
harness phase (2 arms, 50 tasks, concurrency 8) is a few GPU-hours if tasks average around 10 minutes.

## What this does not measure

The engine-side half of the thesis (KV residency across tool waits, pre-filled tool results, speculative next step)
needs a modified vLLM and is the next phase. This directory measures only the harness-side rules under a real model
and real container latencies, which the replay could not do.
