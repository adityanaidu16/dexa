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

_(filled in below as batches complete)_

## Caveats

- These are benchmark tasks in sandboxes, not production sessions; the trace decomposition's timing figures come from production Claude Code and Codex traces and are used here only for the model-time scenarios.
- Test durations depend on the host: 4 vCPUs, one replay at a time, Docker overlay filesystem. Absolute seconds transfer only roughly; the ratios and hit rates transfer better.
- The "edit" that triggers a launch includes creating a brand-new test script, which the agent then runs instead of the previous test; the analysis separates launches by whether the triggering edit created files or modified existing ones.
- A hit's output equality is checked against a real run performed immediately after the speculative one on the same tree; flaky tests would show up as inequality.
