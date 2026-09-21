# Concurrency sweep `20260921T003645Z-sweep-10`

Vanilla arm, one H100, levels run one after the other on the same server. Engine counters are deltas over each level; gauges are sampled every 15 s during it.

| sessions in flight | task runs | errors | submitted | loop median (s) | loop mean (s) | model call median (s) | p90 (s) | stall time (s) | tool median (s) | tasks per GPU-hour, level span | steady state | steady state, stall-free |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 50 | 0 | 34 | 131 | 147 | 1.13 | 7.38 | 177 | 20 | 174 | 196 | 201 |
| 16 | 50 | 0 | 32 | 187 | 218 | 1.64 | 11.72 | 738 | 22 | 231 | 264 | 283 |
| 32 | 100 | 0 | 68 | 222 | 253 | 2.14 | 14.97 | 1331 | 22 | 295 | 455 | 480 |
| 64 | 150 | 0 | 107 | 693 | 742 | 13.55 | 48.50 | 4290 | 21 | 218 | 311 | 323 |

| sessions in flight | model calls | prefix-cache hit rate | uncached prompt tokens per call | preemptions | prefill time (s) | decode time (s) | TTFT mean (ms) | TPOT mean (ms) | generated tokens per s | KV usage mean / max (%) | running requests mean / max | waiting max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1638 | 0.969 | 343 | 0 | 69 | 4147 | 54 | 12.9 | 313 | 0 / 0 | 0.0 / 0 | 0 |
| 16 | 1678 | 0.968 | 373 | 0 | 80 | 6969 | 63 | 19.4 | 462 | 0 / 0 | 0.0 / 0 | 0 |
| 32 | 3188 | 0.965 | 380 | 0 | 170 | 16741 | 74 | 26.8 | 516 | 0 / 0 | 0.0 / 0 | 0 |
| 64 | 4882 | 0.314 | 38669 | 1517 | 2225 | 69235 | 6032 | 72.0 | 392 | 0 / 0 | 0.0 / 0 | 0 |

