# Speculative test execution after edits: a replay experiment

*Status: draft, numbers being filled in as the batch driver finishes. Code and raw records: `experiments/speculative-tool-exec/`.*

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

Replayed so far: **213 sessions, 2,937 tool calls** across 6 repository images.

### 1. How predictable is the command after an edit?

| trajectories | sessions | tool calls | launches, any edit | hit rate, any edit | launches after modifying a file | hit rate | created-file predictions | hit rate, run the new file | speculative output equals real | hit run duration p50 / p90 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| swe-agent | 107 | 1488 | 286 | 31% | 113 | 75% | 165 | 80% | 99% | 1.56 / 1.60 |
| mini-swe-agent | 106 | 1449 | 275 | 32% | 70 | 76% | 200 | 68% | 98% | 0.19 / 0.22 |
| all | 213 | 2937 | 561 | 32% | 183 | 75% | 365 | 73% | 98% | 0.20 / 1.59 |

Two rules cover the post-edit step. **Rule A**, after a call that *modifies* an existing file, launch the most recent test-like command: hit rate 75% over 183 launches. **Rule B**, after a call that *creates* a file, launch that file: hit rate 73% over 365 predictions. Launching the old test after a file creation never hits (185 launches, 1%), which is why a single "rerun the last test" rule measures only 32% across all edits. The `unknown` edit kinds are records from before the edit-kind field was added.

When a speculative run hits, its output matched the output of a real run on the same tree in 98% of cases after normalizing timings and stdout/stderr interleaving; every remaining mismatch inspected was ordering of interleaved streams.

### 2. How long are the runs being overlapped?

In these SWE-smith repositories the speculated runs are short (hit-run duration p50 0.20 s, p90 1.59 s), so the absolute saving inside the benchmark is small. The duration that matters is the production one. From the TraceLab release of real Claude Code and Codex sessions:

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

Per hit, a coding agent on today's model speeds saves about 5 to 8 seconds on a pytest rerun and 3 to 5 on a script rerun; at fast-inference model steps of 1.5 s the saving per hit collapses to about a second, because the overlap window is the model step. The lever pays in proportion to how slow the model is and how slow the tests are, and it is bounded by the number of post-edit reruns per task.

## Caveats

- These are benchmark tasks in sandboxes, not production sessions; the trace decomposition's timing figures come from production Claude Code and Codex traces and are used here only for the model-time scenarios.
- Test durations depend on the host: 4 vCPUs, one replay at a time, Docker overlay filesystem. Absolute seconds transfer only roughly; the ratios and hit rates transfer better.
- The "edit" that triggers a launch includes creating a brand-new test script, which the agent then runs instead of the previous test; the analysis separates launches by whether the triggering edit created files or modified existing ones.
- A hit's output equality is checked against a real run performed immediately after the speculative one on the same tree; flaky tests would show up as inequality.
