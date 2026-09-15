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
4. **The live run confirms the ceiling and fails the gate.** Fifty SWE-bench Verified tasks per arm with an open coder model on one H100 (section below): the rules hit 76 percent of the time, saved 2.3 s per task, and could have saved at most 10.9 s per task with perfect hits and no overhead, against a median task loop of 115 s. Tasks per GPU-hour came out below vanilla on every reading (0.68 to 0.89 against a kill line of 1.15), and the shortfall is server outages and image pulls rather than the rules, but no correction reaches the 1.30 the gate asked for.

**Product reading.** This is a harness feature, not an inference feature: two rules in the agent loop, verifiable by a buyer on their own traces in an afternoon, with a ceiling of a few percent of task time on today's tests. It belongs in an agent SDK or a sandbox product's tool layer, where the sandbox already sees every edit and every command, rather than in a serving engine. The engine-side counterpart, keeping the session's KV resident through the now-overlapped test run, is what turns the same event into a capacity gain; the live run measured the harness half and found it below its kill line, and its token profile (59 prompt tokens per completion token, latency rising with context depth) is the case for measuring the residency half next.

## Phase 1 live: vanilla against the speculative harness on Modal

The live run the replay could not do: an open model drives real tool calls in real sandboxes, once with plain tool
execution and once with the two rules, and the metric is tasks per GPU-hour at equal resolve rate.

**Setup.** `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on vLLM 0.11 on one H100 (Modal), the 50 SWE-bench Verified tasks in
`experiments/speculative-tool-exec/modal/tasks_verified_50.json` (django 20, sympy 10, xarray 5, pytest 5, sphinx 4,
requests 3, pylint 3), bash plus editor plus submit tools, 40 steps and 300 s per command, eight tasks in flight per
arm, arms run one after the other on the same server (`off` first, then `harness`), grading afterwards with the
official SWE-bench eval scripts in fresh sandboxes. Records, logs and the paired analysis are under
`experiments/speculative-tool-exec/modal/runs/20260915T044016Z-harness-7/`. Pre-registered gate: continue at a
harness/off throughput ratio of 1.30 or better at equal resolve rate, kill below 1.15.

**Raw result** (`summary.md`, all 50 tasks per arm):

| metric | off | harness |
|---|---|---|
| tasks | 50 | 50 |
| errored records | 0 | 9 |
| resolved, official grade | 28 | 22 |
| tasks per GPU-hour, arm span | 136 | 94 |
| tasks per GPU-hour, steady state | 158 | 140 |
| wall per task, median (s) | 157 | 146 |
| model time per task, mean (s) | 93 | 135 |
| tool time per task, mean (s) | 29 | 48 |
| sandbox boot, mean (s) | 40.7 | 2.3 |
| speculative launches / hits / misses | 0 | 228 / 162 / 50 |
| saved per task (s) | 0 | 2.3 |

Ratio harness/off: **0.69** by arm span, **0.89** steady state.

**Corrected result** (`analysis.md`, the 41 tasks that completed in both arms, sandbox boot removed):

| metric | off | harness |
|---|---|---|
| resolved, official grade | 24 | 22 |
| agent loop without boot, median (s) | 115 | 143 |
| model time, median (s) | 70 | 102 |
| tool time, median (s) | 19 | 16 |
| tasks per GPU-hour at concurrency 8 | 207 | 142 |

Per-task ratio harness/off: median 1.16, interquartile 1.00 to 1.37. Throughput ratio **0.68**, or **0.77** after
dropping the three slowest tasks in each arm.

**Verdict: kill the harness-side lever.** Every reading of the ratio is below the 1.15 kill line, and the direct
accounting says why no reading could reach 1.30. The rules fired 5.6 times per task and hit 76 percent of the time
(rule B, run the file just created: 143 of 148; rule A, rerun the last test after a modification: 19 of 64), but a hit
saved 0.55 s at the median because the commands being overlapped last 0.9 s. Realized saving: 2.3 s per task. Ceiling
if every launch had hit with zero overhead: 10.9 s per task, 9 percent of a 115 s loop. The mechanics cost about one
extra sandbox round trip per launch, which is the same order as the saving. On this model and this task mix, tool time
is 14 to 17 percent of the loop and the overlappable part of it is a few seconds.

**What the harness arm's deficit is not.** The harness arm is slower than vanilla in every mean, and the paired median
loop is 16 percent longer, but that is not the rules either. Three things are in the number and all are in the records:

1. A server outage at the arm switch. The first eight harness tasks failed on their first model call because the
   gateway returned a non-completion body (nine records with `'str' object has no attribute 'usage'`, a harness bug
   in handling that case, since fixed), and the next eight each waited 164 to 168 s on their first call. The server's own log could not confirm the cause: Modal keeps only the last hundred lines of a stopped app's
   log, and the workflow at the time captured the last three minutes of streaming (it now streams the whole log). The
   signature, eight simultaneous non-completion bodies followed by eight first calls of 164 to 168 s, is consistent
   with the serving container being replaced; the four 258 to 262 s calls inside the off arm are unexplained.
2. Fourteen single model calls over 120 s (3,162 s in total, 12 task runs, both arms), of which the four in the off
   arm each took 258 to 262 s and one harness call took 523 s. They dominate the means; that is why the medians are
   the numbers to read.
3. Sandbox image pulls: the off arm ran first and paid 41 s of image download per task, the harness arm 2 s. The
   corrected table removes boot time; the raw span ratio charges it to the off arm and still lands at 0.69.

None of the three can be spent in the harness arm's favour: with the outage and the boot asymmetry removed the ratio
is 0.68, and the ceiling argument is independent of them.

**Resolve rate.** 24 against 22 on the 41 paired tasks (56 and 54 percent), graded by the official harness with the
gold test patch. The arms are at parity within what three tasks of noise allow; two-thirds of the tasks hit the 40-step
cap without calling submit in both arms.

**What the run says about the other half.** The off arm consumed 17.9 million prompt tokens for 304 thousand
completion tokens, a 59 to 1 ratio, at a demanded 13.5 thousand prompt tokens per second across eight concurrent
sessions on one H100. The per-call model latency rose with context depth, from a median of 0.95 s in the first ten
steps to 1.30 s after step 30, with a p90 of 5.4 to 7.1 s throughout. That is the profile of a server that is
re-prefilling long contexts, which is the tool-aware residency question from the product document: the value on this
workload is not in shaving the 0.9 s test run, it is in whether the 30 to 60 thousand tokens of session context are
still resident when the tool returns. vLLM exposes the numbers that settle it (prefix cache hits and queries,
preemptions, prefill against decode time per request); the harness now snapshots them per arm, and a second run in
interleaved mode (`--spec both`, each task in both arms in the same worker, alternating order) removes the arm-order
and image-cache confounds at the same time.

**Cost.** The run took 64 minutes of H100 time end to end, including grading, roughly $5 at Modal's list price, plus
sandbox CPU.

## Caveats

- These are benchmark tasks in sandboxes, not production sessions; the trace decomposition's timing figures come from production Claude Code and Codex traces and are used here only for the model-time scenarios.
- Test durations depend on the host: 4 vCPUs, one replay at a time, Docker overlay filesystem. Absolute seconds transfer only roughly; the ratios and hit rates transfer better.
- The "edit" that triggers a launch includes creating a brand-new test script, which the agent then runs instead of the previous test; the analysis separates launches by whether the triggering edit created files or modified existing ones.
- A hit's output equality is checked against a real run performed immediately after the speculative one on the same tree; flaky tests would show up as inequality.
- Commands were capped at 300 s inside the container (one hit reached the cap); the first four images were replayed before the edit-kind field existed, so their launches appear as `unknown` and are excluded from the rule A and rule B rates.
- SWE-smith tasks are synthetic bugs injected into real repositories; the agents' reproduce-then-fix loop is the same one seen in the production traces, but task difficulty and test-suite size are not representative of production.
- The mini-SWE-agent corpus does not name its model.
- Live run caveats: one model (Qwen3-Coder-30B-A3B FP8) and one serving stack (vLLM 0.11 on Modal); arms ran in sequence on the same server, so arm order, image-cache state and any server incident fall on one arm; nine harness records errored at the arm switch and are excluded from the paired statistics and counted as unresolved; 14 single model calls over 120 s (3,162 s in total) are in the means and excluded from nothing, which is why the medians are quoted; two-thirds of tasks hit the 40-step cap in both arms.
