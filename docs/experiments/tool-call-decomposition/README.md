# Where the tool time goes: a decomposition of coding-agent tool calls

*Run 2026-09-03 on public traces. No GPU. Scripts and result JSON are in this directory; the datasets are downloaded by the commands in section 6.*

**Question.** If tool calls become the bottleneck for agents, which levers at the inference boundary have room: tool-aware KV residency, a speculative next step during the tool wait, pre-executing a predictable next tool, tool-output compaction, or streaming dispatch? Each lever has a ceiling set by the structure of real traces, and none of those ceilings had been measured.

---

## 1. Data

| corpus | what it is | what it has | what it lacks |
|---|---|---|---|
| TraceLab release v0.0.1 (UW SyFI) | 665,453 LLM rounds in 8,058 Claude Code and Codex sessions from 43 developers; 572,959 tool calls with wall latency. The paper describes a 357k-round snapshot; the release is larger | timestamps for every input event, reasoning, text, and tool call; tool name, wall latency, result size, and the shell executables of each command | all content: prompts, tool inputs, and tool outputs are stripped |
| OpenHands trajectories (Nebius SWE-rebench, Qwen3-Coder-480B) | 400 trajectories, 25,752 tool calls, 185 messages per trajectory on average | full tool arguments and results | timing |
| SWE-agent and mini-SWE-agent (ThoughtWorks corpus; Claude 3.5 and 3.7 Sonnet, GPT-4o, and an unlabeled model for mini) | 600 sessions, 7,733 tool calls, 93% of them shell | full commands and observations | timing |

Timing questions are answered on TraceLab. Structure questions (is the result already known, does the next action depend on it, is the next action predictable) are answered on the two content corpora, which are benchmark tasks in sandboxes rather than production use.

## 2. How a step splits between generation and tools

A step is one LLM invocation plus the tool calls it issues. Generation time is measured from the step's first input event to its last assistant event, so it includes queueing, prefill, decode, and network. Tool time per step is the longest of the step's calls, since a harness issues them in parallel. Human waits (AskUserQuestion, ExitPlanMode) and subagent waits (Agent, TaskOutput, wait_agent) are separated from machine tools.

| | Claude Code | Codex |
|---|---|---|
| steps with at least one machine tool | 261,835 | 173,077 |
| mean generation per step | 16.5 s | 10.3 s |
| mean machine tool time per step | 18.7 s | 8.3 s |
| machine tool share of step wall-clock, time-weighted | 53% | 45% |
| machine tool share at the median step | 3% | 3% |
| steps where machine tool time exceeds generation | 10% | 21% |

Distribution across all 434,912 tool-issuing steps: generation p50 6.6 s, p90 26 s, p99 113 s; tool time p50 0.19 s, p90 11.7 s, p99 216 s.

**Reading.** Tools are 3% of the median step and half of total step time. Both are true because the tail dominates: per TraceLab's own paper, Claude calls under 1 s are 70% of calls and under 1% of tool time. "Tool calls are the bottleneck" is a statement about time, not about steps, and it is a statement about a small number of long calls. It also becomes sharper as generation speeds up: at 5x faster generation, Claude's mean step would be 3.3 s of model time against 18.7 s of tool time, and tools would be 85% of the step.

### 2.1 Where machine tool time goes

| share of all tool time | category | calls | p50 | p90 | p99 | calls over 5 s |
|---|---|---|---|---|---|---|
| 43% | Claude shell (Bash) | 158,385 | 0.33 s | 20 s | 300 s | 20% |
| 31% | Claude human waits | 2,160 | 89 s | 17 min | 9 h | 96% |
| 14% | Codex shell (exec_command, write_stdin) | 236,230 | 0.29 s | 6.2 s | 120 s | 17% |
| 3.8% | Claude subagents | 3,679 | 17 s | 233 s | 773 s | 53% |
| 3.6% | Claude file tools (Read, Edit, Write, Grep, Glob) | 113,172 | 0.04 s | 0.7 s | 31 s | 3% |
| 1.7% | Claude web | 5,345 | 7.7 s | 48 s | 248 s | 75% |

Within machine tool time, the largest single identities are Codex `write_stdin` at 16% (the harness polling a running process, p50 5.0 s), Claude `Bash: tail` at 11%, and Claude `Bash: sleep` at 5%. About a third of machine tool time is therefore the agent waiting on something already running, which it does by polling. Test-like and build-like executables (pytest, python scripts, npm, cargo, make, docker, conda, and similar) account for 40% of shell time.

### 2.2 Idle windows: what the KV cache waits through

The idle window is the time from a step's last assistant event to the next round's start, on rounds triggered by a tool result rather than a user message.

| machine tool-triggered idle windows | value |
|---|---|
| count | 416,372 |
| p50 / p90 / p99 | 0.19 s / 12.3 s / 176 s |
| share of windows over 1 s / 5 s / 25 s / 60 s | 36% / 19% / 7% / 3.5% |
| share of idle *time* in windows over 1 s / 5 s / 25 s / 60 s | 99% / 96% / 87% / 77% |
| human-triggered windows, p50 / p90 / p99 | 61 s / 12.5 min / 5.7 h |

**Reading.** The distribution is bimodal in effect. Four fifths of tool waits are under 5 s and cost nothing to hold in HBM. One fifth exceed 5 s and hold 96% of all idle time; 7% exceed 25 s and hold 87%. A residency policy only has to be right about that fifth.

### 2.3 Can the engine tell a long wait from the tool identity alone?

Identity is the tool name plus the first non-trivial shell executable (1,121 identities). A wait is "long" if it exceeds 5 s; 14% of machine calls are long. Predicting long when the identity's historical long-rate is at least 50%:

| metric | value |
|---|---|
| accuracy | 89% |
| precision on long | 61% |
| recall on long | 57% |
| share of long-wait *time* correctly predicted | 33% |

**Reading.** Tool identity alone is not a good enough hint: it catches a third of the long-wait time. The harness knows more than the identity: the timeout it set, whether the command is a test or build, whether it is polling a background process. Those are the fields a residency hint has to carry, and they are free for the harness to send. The polling third is the easy case: a session that just called `sleep 30` or `write_stdin` with a yield time has announced its own idle window.

## 3. Structure: what the next step needs from the result

Measured on the content corpora. "Reconstructible" means the result was already determined by context: a file view of content unchanged since it was last viewed or edited, an editor's echo after a successful edit, a repeated command with no intervening write, or an acknowledgment. "Uses new information" means the next action's arguments contain a token that first appeared in the immediately preceding result; this is a lower bound on dependence, because an agent can depend on a result without copying from it.

| metric | OpenHands, Qwen3-Coder | SWE-agent family |
|---|---|---|
| tool calls | 25,752 | 7,733 |
| read-only calls | 48% | 36% |
| result reconstructible from prior context, by calls | 29% | 4% |
| reconstructible, by result characters | 10% | 3% |
| file views that re-read unchanged content | 33% of views (5% of all result characters) | 27% of views |
| result is an exact duplicate of an earlier result | 7% | 14% |
| next action copies a new token from the result | 2.6% (views: 6%) | 9.3% (views: 36%) |
| next action is an exact repeat of an earlier action | 8.9% | 10% |

### 3.1 Does the next action depend on whether a command failed?

Next action type after a shell command, by outcome (failure detected from tracebacks, error markers, and non-zero exits in the observation):

| next action | OpenHands, ok | OpenHands, failed | SWE-agent, ok | SWE-agent, failed |
|---|---|---|---|---|
| another shell command | 54% | 52% | 96% | 93% |
| a file view | 25% | 16% | 1.6% | 0.6% |
| an edit | 16% | 25% | 2.3% | 6.0% |

The type of the next action shifts only modestly with failure. The content of the next action is not measurable this way, so this is evidence that a speculative next step would often be the right *kind* of step, not that it would be the right step.

### 3.2 The one predictable expensive step: the rerun after an edit

| after a `str_replace` edit | OpenHands | SWE-agent family |
|---|---|---|
| next action is a shell command | 59% | 79% |
| next action is an exact repeat of an earlier command | 50% | 74% |
| next command is a test or script run | 56% | 77% |

The repeated commands are almost entirely `cd <repo> && python reproduce*.py` and `pytest ...`: the agent re-runs its reproduction script after every edit. Repeated commands are otherwise rare (18 and 9 cases without an intervening write, and their output matched only a third of the time).

## 4. What this says about each lever

**Tool-aware residency.** Room exists and the shape is favorable: 81% of waits need no decision, and the 19% that do hold 96% of idle time. The engine cannot make the decision from the tool name (33% of long-wait time caught), so the product is a hint protocol the harness fills in, and a third of long waits are announced polls. The claim to prove is agent steps per GPU-hour at a fixed resume latency, with the hint versus without. Unmeasured: the resume cost at 120k-token contexts under concurrency, which the session-candidate verifiers already flagged as the open number.

**Pre-executing the rerun after an edit.** This is the measured win. Half to three quarters of post-edit steps re-run a known command, that command is a test or reproduction script, and test-like commands are 40% of shell time. A harness can launch the last test command the instant an edit lands and hand the model its result when it asks, overlapping the run with the model's own next decode. Hit rate 50 to 74% on these corpora; the saved time is the test's wall time. This is a harness or SDK feature, verifiable on the buyer's own traces, and it is the cheapest thing in this document to build.

**Speculative next step inside the engine.** The upper bound is high (the next action's type barely depends on the outcome and rarely copies from the result) but the measured lower bound is the exact-repeat rate, 9 to 10% of all steps, concentrated in the post-edit rerun above. Outside that pattern the next action's content is not predictable from the trace by any method tested here. Building the general mechanism before measuring a learned predictor's hit rate would be betting on the gap between 10% and the upper bound.

**Tool-output compaction and re-read elimination.** Re-reads of unchanged files are a third of views in the OpenHands agent but 5% of result characters; reconstructible results are 10% of characters. The token savings from eliminating known results are real but moderate on these agents; compaction's value has to come from the size distribution of long outputs, which this analysis did not measure.

**Streaming dispatch.** Not measurable here; it saves the decode tail of the tool call, and the median generation of 6.6 s bounds what it can save.

## 5. Caveats

- Generation time on TraceLab includes queueing and network; timestamps are event emission times. The release is larger than the paper's snapshot, so numbers differ from the paper's tables.
- The content corpora are benchmark tasks in sandboxes with fixed agent scaffolds; production agents with human turns and richer tools may differ. Structure numbers are per corpus and were not weighted by task outcome.
- "Uses new information" is a token-copy proxy and a lower bound on dependence. Write detection in shell commands is a conservative regular expression; anything that could write invalidates known file state.
- Long-wait predictability uses in-sample identity rates, which flatters the predictor slightly.

## 6. Reproduce

```
curl -L --fail -o syfi_coding_trace.jsonl.gz https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.jsonl.gz
python3 analyze_tracelab2.py                      # timing decomposition, idle windows, predictability
# content corpora, 400 and 600 rows streamed from Hugging Face:
#   nebius/SWE-rebench-openhands-trajectories -> nebius_400.json
#   thoughtworks/agentic-coding-trajectories  -> thoughtworks_600.json ; python3 adapt_thoughtworks.py
python3 analyze_content.py nebius_400.json nebius_structure.json
python3 analyze_content2.py nebius_400.json nebius_control.json
```

## 7. The next experiment

No GPU is needed for the measured win. Modify one open harness (mini-SWE-agent or OpenHands) to launch the most recent test command the moment an edit succeeds, then run a 50-task SWE-bench Verified subset with and without it on the same model and report wall-clock per task, dollars per task, and any change in resolve rate. That is about two days of work and a few hundred dollars of API spend, and it turns the 50 to 74% hit rate into a measured minutes-per-task number a buyer can rerun.

The residency lever needs the GPU experiment the decision document already specifies, with one addition: the load harness replays these idle-window distributions and carries the tool hint.
