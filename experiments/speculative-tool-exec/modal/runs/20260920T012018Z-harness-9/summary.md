| metric | harness | off |
|---|---|---|
| tasks | 50 | 50 |
| errors | 0 | 0 |
| resolved (official grade) | 27 | 24 |
| submitted | 30 | 33 |
| concurrency | 8 | 8 |
| tasks per GPU-hour (arm span) | 57.53 | 73.58 |
| tasks per GPU-hour (steady state, no tail) | 144.29 | 166.10 |
| arm span (s) | 3129 | 2446 |
| wall per task, mean (s) | 200 | 174 |
| wall per task, median (s) | 134 | 141 |
| agent loop per task, mean (s) | 200 | 173 |
| model time per task (s) | 135 | 116 |
| tool time per task (s) | 38 | 35 |
| sandbox boot (s) | 2.1 | 2.1 |
| steps per task | 32.2 | 33.5 |
| prompt tokens per task | 348042 | 400660 |
| completion tokens per task | 6655 | 7485 |
| speculative launches | 269 | 0 |
| hits | 197 | 0 |
| misses | 61 | 0 |
| hit rate | 0.76 | 0.00 |
| saved per task (s) | 2.1 | 0.0 |
| wasted per task (s) | 26.5 | 0.0 |

interleaved run: both arms shared the server for the whole span, so arm-span throughput is not meaningful and the absolute steady-state figures assume each arm had the server alone; the ratio is the number to read. harness / off ratio of summed agent loop: 0.869  (gate: >= 1.30 continue, < 1.15 kill; judged only at equal resolve rate; see analysis.md for the paired, boot-corrected version)
