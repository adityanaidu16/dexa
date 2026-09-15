| metric | harness | off |
|---|---|---|
| tasks | 3 | 3 |
| errors | 0 | 0 |
| resolved (official grade) | 1 | 2 |
| submitted | 1 | 2 |
| concurrency | 1 | 1 |
| tasks per GPU-hour (arm span) | 36.64 | 29.50 |
| tasks per GPU-hour (steady state, no tail) | 36.94 | 29.69 |
| arm span (s) | 295 | 366 |
| wall per task, mean (s) | 98 | 122 |
| wall per task, median (s) | 92 | 94 |
| agent loop per task, mean (s) | 97 | 121 |
| model time per task (s) | 55 | 82 |
| tool time per task (s) | 16 | 15 |
| sandbox boot (s) | 1.9 | 1.8 |
| steps per task | 38.3 | 37.0 |
| prompt tokens per task | 329718 | 441338 |
| completion tokens per task | 6029 | 9199 |
| speculative launches | 12 | 0 |
| hits | 1 | 0 |
| misses | 9 | 0 |
| hit rate | 0.10 | 0.00 |
| saved per task (s) | 0.2 | 0.0 |
| wasted per task (s) | 3.0 | 0.0 |

harness / off throughput ratio: 1.242 by arm span, 1.244 steady state  (gate: >= 1.30 continue, < 1.15 kill; judged only at equal resolve rate)
