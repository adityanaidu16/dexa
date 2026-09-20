# Speculative test execution after edits: a replay experiment

*Final. Code, raw per-session records, and the aggregation script: `experiments/speculative-tool-exec/`. Replayed 2026-09-03 to 2026-09-04; live phase-1 run on Modal 2026-09-15, records under `experiments/speculative-tool-exec/modal/runs/`.*

## Question

The trace decomposition (`tool-call-decomposition.md`) found that after a `str_replace` edit, coding agents' next action is a shell command 59 to 79% of the time and an exact repeat of an earlier command 50 to 74% of the time, almost always the reproduction script or the test run. If a harness launched that command the instant the edit landed, the model's next decode step and the test run would overlap. This experiment measures, in real task environments, how often that speculation would hit, whether the speculative output is the same output the agent would have seen, how long the overlapped runs take, and what a live agent would therefore save.

## Method

**Replay, not re-generation.** The policy does not change what the model sees when it hits (same command, same tree state, same output), so the model's decisions are unchanged and a recorded trajectory can be replayed action by action inside the task's real Docker image. No LLM calls are made; every recorded tool call is executed for real and timed.

**Policy under test.** After any tool call that changes the working tree, if the most recent test-like shell command is known and safe (no redirects, deletes, installs, or git state changes), launch it immediately in the background inside the container. When the trajectory's next state-changing action arrives:

- if it is that same command (exact text, or equal after dropping a leading `cd .` / `cd /testbed` hop and collapsing whitespace), count a **hit**: the output is already computed;
- anything else kills the speculative run and counts a **miss**;
- read-only commands in between (`cat`, `ls`, `grep`, `git diff`, editor `view`) do not disturb a pending run.

For every hit the command is also run for real afterwards and the two outputs compared, exactly and after normalizing timings, addresses, and stdout/stderr interleaving. The speculative run's duration `D` is recorded; a live agent saves `min(D, M)` where `M` is the model time of the next step, reported here for `M` in {1.5 s, 6.6 s, 14.2 s, 26.2 s}, the fast-inference case and the TraceLab p50, mean, and p90 of per-step generation time.

**Environments.** SWE-smith task images (`jyangballin/swesmith.x86_64.*`), one per repository snapshot, with the task's bug patch applied before replay. The harness helpers the recorded agents relied on (`str_replace_editor` for SWE-agent, `edit_via_str_replace` for mini-SWE-agent) are re-implemented and installed into each container.

**Trajectories.** The ThoughtWorks agentic-coding-trajectories corpus: SWE-agent sessions recorded with Claude 3.7 Sonnet, Claude 3.5 Sonnet, and GPT-4o, and mini-SWE-agent sessions from the Klear SWE-smith set (model unspecified in the corpus). Sessions were selected by image so that each pulled image serves many sessions.

## Results

Replayed so far: **480 sessions, 6,628 tool calls** across 21 repository images.

### 1. How predictable is the command after an edit?

| trajectories | sessions | tool calls | launches, any edit | hit rate, any edit | launches after modifying a file | hit rate | created-file predictions | hit rate, run the new file | speculative output equals real | hit run duration p50 / p90 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| swe-agent | 201 | 2799 | 554 | 33% | 246 | 73% | 331 | 76% | 98% | 0.64 / 1.63 |
| mini-swe-agent | 279 | 3829 | 817 | 31% | 278 | 78% | 678 | 89% | 97% | 0.30 / 1.60 |
| all | 480 | 6628 | 1371 | 32% | 524 | 76% | 1009 | 85% | 98% | 0.31 / 1.63 |

Two rules cover the post-edit step. **Rule A**, after a call that *modifies* an existing file, launch the most recent test-like command: hit rate 76% over 524 launches. **Rule B**, after a call that *creates* a file, launch that file: hit rate 85% over 1009 predictions. Launching the old test after a file creation never hits (578 launches, 1%), which is why a single "rerun the last test" rule measures only 32% across all edits. The `unknown` edit kinds are records from before the edit-kind field was added.

When a speculative run hits, its output matched the output of a real run on the same tree in 98% of cases after normalizing timings and stdout/stderr interleaving; every remaining mismatch inspected was ordering of interleaved streams.

### 2. How long are the runs being overlapped?

In these SWE-smith repositories the speculated runs are short (hit-run duration p50 0.31 s, p90 1.63 s), so the absolute saving inside the benchmark is small. The duration that matters is the production one. From the TraceLab release of real Claude Code and Codex sessions:

| production tool | calls | p50 (s) | p90 (s) | p99 (s) | share over 5 s | time in calls over 5 s |
|---|---|---|---|---|---|---|
| claude pytest | 1,460 | 9.2 | 77.0 | 183 | 65% | 97% |
| claude python | 28,396 | 2.1 | 47.0 | 563 | 32% | 98% |
| claude build-tool | 2,162 | 7.4 | 46.3 | 303 | 59% | 98% |
| codex pytest | 4,993 | 1.2 | 5.2 | 30 | 12% | 65% |
| codex python | 37,646 | 1.1 | 4.9 | 30 | 10% | 88% |

### 3. What a live agent would save per hit

A hit saves `min(D, M)`: the test's duration `D`, capped by the model time `M` of the next step it overlaps. Taking `D` from the production Claude Code distribution above and `M` from TraceLab's per-step generation time:

| hit on a ... | mean duration (s) | saved at model step 1.5 s | 6.6 s (p50) | 14.2 s (mean) | 26.2 s (p90) |
|---|---|---|---|---|---|
| Claude Code pytest run | 26.4 | 1.4 | 4.9 | 8.5 | 12.1 |
| Claude Code python run | 42.4 | 1.0 | 2.9 | 4.6 | 6.5 |

Under the two rules, the replayed sessions contain on average **2.6 predictable post-edit runs per session** (398 rule-A hits plus 856 rule-B hits over 480 sessions), so the per-task saving is that count times the per-hit figure below.

Per hit, a coding agent on today's model speeds saves about 5 to 8 seconds on a pytest rerun and 3 to 5 on a script rerun; at fast-inference model steps of 1.5 s the saving per hit collapses to about a second, because the overlap window is the model step. The lever pays in proportion to how slow the model is and how slow the tests are, and it is bounded by the number of post-edit reruns per task.

## Conclusions

1. **The post-edit action is predictable enough to pre-execute.** Two rules cover it. After a modification of an existing file, rerun the most recent test-like command: 398 of 524 launches hit (76%; 73% on the Claude 3.5/3.7 and GPT-4o sessions, 78% on the mini-SWE-agent sessions). After a file creation, run the created file: 856 of 1009 (85%). The rule the trace decomposition suggested on its own, rerun the last test after any edit, scores only 32% because after creating a new script the agent runs the new script, not the old test (0.9% over 578 launches).
2. **A hit is safe.** In 97.5% of 442 hits the speculative output equalled a real run on the same tree; the inspected remainder differed only in stdout/stderr interleaving. Speculating on read-only tools in between costs nothing, and a miss wastes 0.5 s of container CPU on average.
3. **The saving is set by the test, not by the harness.** Sessions contain 2.6 predictable post-edit runs on average. Inside these SWE-smith repositories the runs last 0.3 s at the median, so the benchmark itself saves seconds per task. On the production distribution the same hit is worth roughly 5 s on a pytest rerun and 3 s on a script rerun at today's median model step, rising to 12 s and 6 s at the p90 step, and falling to about a second if model steps drop to 1.5 s. Per task that is tens of seconds today against a median task of several minutes, and it shrinks as inference gets faster, which is the opposite of the tool-aware residency lever, whose value grows as inference gets faster.
4. **The live run confirms the ceiling and fails the gate.** Fifty SWE-bench Verified tasks, each run in both arms with an open coder model on one H100 (section below): the rules hit 76 percent of the time, saved 2.1 s per task against a median loop of 138 s, and the arms came out at parity (throughput ratio 0.99 with transport stalls removed, typical task 3 percent slower with speculation on, resolve 27 against 24) against a kill line of 1.15. The server's counters add the finding that at eight agent sessions per GPU the prefix cache already holds 97 percent of every prompt with no evictions, so the residency lever has nothing to work on until sessions per GPU rise; the concurrency sweep that finds that point is the next run.

**Product reading.** This is a harness feature, not an inference feature: two rules in the agent loop, verifiable by a buyer on their own traces in an afternoon, with a ceiling of a few percent of task time on today's tests. It belongs in an agent SDK or a sandbox product's tool layer, where the sandbox already sees every edit and every command, rather than in a serving engine. The engine-side counterpart, keeping the session's KV resident through the now-overlapped test run, is what turns the same event into a capacity gain; the live run measured the harness half and found it at parity, and its server counters (97 percent prefix-cache hit rate, no preemptions, prefill under 2 percent of engine time at eight sessions per GPU) say the residency half only starts to matter at higher session density, which is the next measurement.

## Phase 1 live: vanilla against the speculative harness on Modal

The live run the replay could not do: an open model drives real tool calls in real sandboxes, once with plain tool
execution and once with the two rules, and the metric is tasks per GPU-hour at equal resolve rate.

**Setup.** `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on vLLM 0.11 on one H100 (Modal), the 50 SWE-bench Verified tasks in
`experiments/speculative-tool-exec/modal/tasks_verified_50.json` (django 20, sympy 10, xarray 5, pytest 5, sphinx 4,
requests 3, pylint 3), bash plus editor plus submit tools, 40 steps and 300 s per command, eight tasks in flight,
grading afterwards with the official SWE-bench eval scripts in fresh sandboxes. Pre-registered gate: continue at a
harness/off throughput ratio of 1.30 or better at equal resolve rate, kill below 1.15. Two runs: run 7 ran the arms one
after the other and was confounded (below); run 9 ran every task in both arms inside the same worker, alternating which
arm went first, so both arms share server state, image cache and any incident, and it is the result. Records, logs, the
vLLM metrics snapshot and the paired analysis are under `experiments/speculative-tool-exec/modal/runs/`.

**Result, run 9** (`20260920T012018Z-harness-9`, 50 paired tasks, no errors):

| metric | off | harness |
|---|---|---|
| resolved, official grade | 24 | 27 |
| agent loop per task, median (s) | 138 | 132 |
| agent loop per task, mean (s) | 171 | 197 |
| model time per task, median (s) | 92 | 82 |
| tool time per task, median (s) | 20 | 17 |
| steps per task, mean | 33.5 | 32.2 |
| speculative launches / hits / misses | 0 | 269 / 197 / 61 |
| saved per task (s) | 0 | 2.1 |

Throughput ratio harness/off from the summed loop time (the only throughput figure that means anything when both arms
share the server): **0.87 raw**, **0.99 with stall time removed** (the part of any single model call beyond 120 s, see
below). Per task, the harness loop over the vanilla loop has median **1.03** with interquartile 0.82 to 1.24, so the
typical task is 3 percent slower with speculation on. Order check on stall-free per-task loop ratios: 1.08 when the
harness arm ran a task first, 0.97 when the vanilla arm did; the second run of a task is 7 percent faster than its
first whichever arm it is, and alternating the order cancels it. Dropping the three slowest tasks in each arm: 0.95.

**Verdict: kill the harness-side lever.** The arms are at parity. The rules fired 5.4 times per task and hit 76 percent
of the time (rule B, run the file just created: 175 of 190; rule A, rerun the last test after a modification: 22 of
68), the same rates as run 7 and close to the replay's, but a hit saved 0.53 s because the commands being overlapped
last 0.93 s at the median. Realized saving: 2.1 s per task against a median loop of 138 s. The mechanics cost about one
sandbox round trip per launch, the same order as the saving. Resolve rate 27 against 24 is inside the noise of 50
tasks graded once. Tool time is 13 to 15 percent of the loop on this model and task mix and the overlappable slice of
it is a few seconds; no correction reaches 1.15, let alone 1.30.

**What the server was doing** (`metrics_summary.md`, vLLM's own counters over the 3,289 model calls of run 9):

| metric | value |
|---|---|
| prompt tokens per call, mean | 11,401 |
| generation tokens per call, mean | 216 |
| prefix-cache hit rate | 97.0 percent (345 uncached prompt tokens per call) |
| preemptions | 0 |
| prefill time, all calls | 155 s |
| decode time, all calls | 8,599 s |
| time to first token, mean | 59 ms |
| time per output token, mean | 12.2 ms |
| slowest call seen by the server | under 60 s (6 calls between 50 and 60 s) |
| KV cache | 452,512 tokens; eight sessions of 11 to 50 thousand tokens fit |

Three readings follow. First, at eight agent sessions per H100 there is no residency problem to solve: 97 percent of
every prompt is already resident, prefill is under 2 percent of engine time, nothing is ever evicted. The engine is
decode-bound and mostly idle, serving two to three active decodes on average while the other sessions are in tool
calls, sandbox round trips or thinking. The residency lever, which the product document rates as the engine-side
half of this thesis, only has something to work on once sessions per GPU rise until the KV budget is exceeded; how far
that is, and what tasks per GPU-hour do on the way, is a concurrency sweep on the vanilla arm and is the natural next
run. Second, the per-call latency the agent sees rises with context depth (median 0.97 s in the first ten steps to
1.41 s after step 30, p90 6 to 8 s) with time to first token flat at 59 ms, so that growth is decode over a longer
KV, not re-prefill. Third, the stalls are not the engine's. Five model calls in 100 task runs took 311 to 910 s on
the client side, 3,368 s in total or 8 percent of all loop time, three of them exactly the client's 900 s timeout
followed by a fast retry, while the server's own histogram shows no call over 60 s. They are lost between the runner
and the engine, in the HTTP ingress or the connection, and any agent-serving product has to design for them
(streaming, keepalive, short timeouts with retry); the harness now uses a 300 s timeout with three retries.

**Run 7, the first attempt** (`20260915T044016Z-harness-7`, arms in sequence, vanilla first): ratio 0.69 by arm span
and 0.68 on 41 paired tasks with boot removed. It was confounded three ways, all visible in its records: an outage at
the arm switch cost the harness arm nine errored tasks and eight first calls of 165 s each; the vanilla arm, running
first, paid 41 s per task of sandbox image pulls; and 14 calls stalled over 120 s, unevenly. Its hit rates (76 percent,
rule B 97, rule A 30), saved time (2.3 s per task) and resolve rates (24 against 22 on the paired tasks) match run 9,
which is what the interleaved design was built to check.

**Cost.** Run 9 took 65 minutes of H100 time end to end including grading, about $5 at list price plus sandbox CPU;
all runs together about three H100-hours.

## Caveats

- These are benchmark tasks in sandboxes, not production sessions; the trace decomposition's timing figures come from production Claude Code and Codex traces and are used here only for the model-time scenarios.
- Test durations depend on the host: 4 vCPUs, one replay at a time, Docker overlay filesystem. Absolute seconds transfer only roughly; the ratios and hit rates transfer better.
- The "edit" that triggers a launch includes creating a brand-new test script, which the agent then runs instead of the previous test; the analysis separates launches by whether the triggering edit created files or modified existing ones.
- A hit's output equality is checked against a real run performed immediately after the speculative one on the same tree; flaky tests would show up as inequality.
- Commands were capped at 300 s inside the container (one hit reached the cap); the first four images were replayed before the edit-kind field existed, so their launches appear as `unknown` and are excluded from the rule A and rule B rates.
- SWE-smith tasks are synthetic bugs injected into real repositories; the agents' reproduce-then-fix loop is the same one seen in the production traces, but task difficulty and test-suite size are not representative of production.
- The mini-SWE-agent corpus does not name its model.
- Live run caveats: one model (Qwen3-Coder-30B-A3B FP8) and one serving stack (vLLM 0.11 on Modal); 50 tasks graded once per arm, so resolve-rate differences of a few tasks are noise; the harness's own bookkeeping (a tree fingerprint after every state-changing call) costs 0.66 s per step in both arms and inflates absolute loop times by about 21 s per task; five client-side stalls of 311 to 910 s are in the raw means and removed in the stall-free figures; two-thirds of tasks hit the 40-step cap in both arms; run 7's sequential-arm confounds are described in its paragraph.
