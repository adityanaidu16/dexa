| metric | harness | off |
|---|---|---|
| tasks | 50 | 50 |
| errors | 9 | 0 |
| resolved (official grade) | 22 | 28 |
| submitted | 26 | 37 |
| concurrency | 8 | 8 |
| tasks per GPU-hour (arm span) | 94.39 | 136.06 |
| tasks per GPU-hour (steady state, no tail) | 139.97 | 157.91 |
| arm span (s) | 1907 | 1323 |
| wall per task, mean (s) | 206 | 183 |
| wall per task, median (s) | 146 | 157 |
| agent loop per task, mean (s) | 206 | 182 |
| model time per task (s) | 135 | 93 |
| tool time per task (s) | 48 | 29 |
| sandbox boot (s) | 2.3 | 40.7 |
| steps per task | 31.9 | 32.8 |
| prompt tokens per task | 377695 | 358237 |
| completion tokens per task | 6294 | 6074 |
| speculative launches | 228 | 0 |
| hits | 162 | 0 |
| misses | 50 | 0 |
| hit rate | 0.76 | 0.00 |
| saved per task (s) | 2.3 | 0.0 |
| wasted per task (s) | 5.9 | 0.0 |

harness / off throughput ratio: 0.694 by arm span, 0.886 steady state  (gate: >= 1.30 continue, < 1.15 kill; judged only at equal resolve rate)
