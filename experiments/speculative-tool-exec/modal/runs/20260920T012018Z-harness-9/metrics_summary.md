# vLLM metrics at the end of `20260920T012018Z-harness-9`

| metric | value |
|---|---|
| requests | 3289 |
| prompt tokens | 37498825 (11401 per request) |
| generation tokens | 709473 (216 per request) |
| prefix cache hit rate | 0.9697 (1136297 uncached prompt tokens, 345 per request) |
| preemptions | 0 |
| prefill time, sum (s) | 155 |
| decode time, sum (s) | 8599 |
| queue time, sum (s) | 0.29 |
| time to first token, mean (ms) | 59 |
| time per output token, mean (ms) | 12.2 |
| end-to-end latency, mean (s) | 2.67 |

KV cache: 28282 blocks of 16 tokens = 452,512 tokens; prefix caching True; gpu_memory_utilization 0.9.

## End-to-end request latency

| bucket | requests |
|---|---|
| up to 0.3 s | 72 |
| up to 0.5 s | 579 |
| up to 0.8 s | 683 |
| up to 1 s | 413 |
| up to 1.5 s | 321 |
| up to 2 s | 157 |
| up to 2.5 s | 145 |
| up to 5 s | 375 |
| up to 10 s | 381 |
| up to 15 s | 111 |
| up to 20 s | 30 |
| up to 30 s | 9 |
| up to 40 s | 1 |
| up to 50 s | 6 |
| up to 60 s | 6 |

## Prompt tokens per request

| bucket | requests |
|---|---|
| up to 1000 tok | 78 |
| up to 2000 tok | 156 |
| up to 5000 tok | 343 |
| up to 10000 tok | 932 |
| up to 20000 tok | 1442 |
| up to 50000 tok | 338 |

## Time to first token

| bucket | requests |
|---|---|
| up to 0.02 s | 82 |
| up to 0.04 s | 1440 |
| up to 0.06 s | 37 |
| up to 0.08 s | 966 |
| up to 0.1 s | 635 |
| up to 0.25 s | 119 |
| up to 0.5 s | 1 |
| up to 0.75 s | 3 |
| up to 1 s | 2 |
| up to 2.5 s | 4 |
