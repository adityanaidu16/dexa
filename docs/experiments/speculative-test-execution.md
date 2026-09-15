# Speculative test execution after edits: a replay experiment

*Final. Code, raw per-session records, and the aggregation script: `experiments/speculative-tool-exec/`. Replayed 2026-09-03 to 2026-09-04.*

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
4. **What this does not show.** A live agent was not run: the replay holds the trajectory fixed and asks only whether the prediction would have been right and whether its output would have matched. The live harness in `experiments/speculative-tool-exec/live_agent.py` implements the same two rules with the Anthropic SDK against SWE-bench Verified images and reports minutes and dollars per task with speculation on and off; it needs an API key in the environment.

**Product reading.** This is a harness feature, not an inference feature: two rules in the agent loop, verifiable by a buyer on their own traces in an afternoon, with a ceiling of a few percent of task time on today's tests. It belongs in an agent SDK or a sandbox product's tool layer, where the sandbox already sees every edit and every command, rather than in a serving engine. The engine-side counterpart, keeping the session's KV resident through the now-overlapped test run, is what turns the same event into a capacity gain, and that is the unmeasured half.

## Caveats

- These are benchmark tasks in sandboxes, not production sessions; the trace decomposition's timing figures come from production Claude Code and Codex traces and are used here only for the model-time scenarios.
- Test durations depend on the host: 4 vCPUs, one replay at a time, Docker overlay filesystem. Absolute seconds transfer only roughly; the ratios and hit rates transfer better.
- The "edit" that triggers a launch includes creating a brand-new test script, which the agent then runs instead of the previous test; the analysis separates launches by whether the triggering edit created files or modified existing ones.
- A hit's output equality is checked against a real run performed immediately after the speculative one on the same tree; flaky tests would show up as inequality.
- Commands were capped at 300 s inside the container (one hit reached the cap); the first four images were replayed before the edit-kind field existed, so their launches appear as `unknown` and are excluded from the rule A and rule B rates.
- SWE-smith tasks are synthetic bugs injected into real repositories; the agents' reproduce-then-fix loop is the same one seen in the production traces, but task difficulty and test-suite size are not representative of production.
- The mini-SWE-agent corpus does not name its model.
