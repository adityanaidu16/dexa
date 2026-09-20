# Paired analysis of `20260920T012018Z-harness-9`

Tasks: 50 off, 50 harness. Errored records: 0.

## Paired tasks (n=50)

| metric | off | harness |
|---|---|---|
| resolved (official grade) | 24 | 27 |
| agent loop minus sandbox boot, median (s) | 138 | 132 |
| agent loop minus sandbox boot, mean (s) | 171 | 197 |
| sandbox boot, mean (s) | 2.1 | 2.1 |
| model time, median (s) | 92 | 82 |
| tool time, median (s) | 20 | 17 |
| steps, mean | 33.5 | 32.2 |

Per-task ratio harness/off of the boot-corrected loop: median 1.029, IQR 0.82 to 1.24.
Tasks per GPU-hour on the paired tasks at concurrency 8, boot excluded: off 168, harness 146, ratio 0.868.
Interleaved run: the absolute figures assume each arm had the server alone; the ratio is the number to read. Order check, ratio when harness ran first (n=25): 1.701; when off ran first (n=25): 0.820.
Dropping the 3 slowest tasks in each arm: ratio 0.948.
Removing the part of any single model call beyond 120 s (stalls): sum ratio 0.986, per-task median 1.029.
Order check on medians, stall-free: harness ran first (n=25) median ratio 1.079; off ran first (n=25) 0.969. Second run of a task against its first run, any arm: median 0.926.

## Speculation accounting (harness arm, paired tasks)

- launches 269 (5.4 per task), hits 197, misses 61, hit rate 0.76
- rule A: 22 hits, 46 misses, hit rate 0.32
- rule B: 175 hits, 15 misses, hit rate 0.92
- saved per hit: median 0.53 s, mean 0.53 s, max 0.8 s; speculative command duration median 0.93 s
- saved per task 2.10 s; killed-run time per task 26.54 s (background, not wall clock); launch/poll/kill round trips are extra sandbox execs of roughly 0.3 to 0.5 s each, not itemized
- ceiling: if every launch had hit with zero overhead, saving would be at most 30.7 s per task against a median loop of 138 s

## Model latency by context depth (both arms pooled)

| steps | calls | median (s) | p90 (s) | max (s) |
|---|---|---|---|---|
| 0 to 9 | 1000 | 0.97 | 6.35 | 904 |
| 10 to 19 | 979 | 1.01 | 7.52 | 337 |
| 20 to 29 | 809 | 1.31 | 8.37 | 311 |
| 30 to 39 | 496 | 1.41 | 8.02 | 910 |

Stalls (single model calls over 120 s): 5, total 3368 s, across 5 task runs: django__django-11087 (off, step 0, 904 s), django__django-10973 (harness, step 20, 311 s), django__django-11095 (harness, step 37, 910 s), django__django-11179 (harness, step 13, 337 s), pytest-dev__pytest-10051 (harness, step 34, 907 s)

Off arm token totals: 20.0M prompt, 374k completion, ratio 54:1.

## Per-task table

| instance | off loop (s) | off model | off tool | off steps | off resolved | harness loop (s) | model | tool | steps | resolved | hits | misses | saved (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| django__django-11095 | 205 | 150 | 22 | 37 | True | 1151 | 1099 | 15 | 40 | True | 6 | 1 | 3.1 |
| pytest-dev__pytest-10081 | 735 | 96 | 623 | 40 | True | 1028 | 63 | 923 | 40 | False | 3 | 9 | 1.3 |
| pytest-dev__pytest-10051 | 84 | 40 | 21 | 34 | False | 1026 | 989 | 16 | 40 | False | 4 | 3 | 2.0 |
| django__django-10973 | 228 | 121 | 43 | 36 | False | 467 | 409 | 23 | 33 | True | 4 | 1 | 1.9 |
| django__django-11179 | 472 | 408 | 26 | 39 | True | 424 | 390 | 14 | 25 | True | 3 | 1 | 1.5 |
| django__django-11163 | 152 | 99 | 27 | 28 | True | 305 | 268 | 17 | 22 | True | 1 | 4 | 0.6 |
| sphinx-doc__sphinx-10435 | 163 | 115 | 31 | 35 | False | 228 | 189 | 18 | 40 | False | 4 | 1 | 2.2 |
| sphinx-doc__sphinx-10323 | 119 | 80 | 21 | 27 | True | 206 | 162 | 18 | 40 | True | 9 | 3 | 5.1 |
| django__django-10880 | 164 | 111 | 21 | 35 | False | 177 | 130 | 18 | 40 | True | 4 | 1 | 2.0 |
| django__django-11087 | 1034 | 1009 | 10 | 29 | False | 176 | 115 | 31 | 38 | False | 0 | 5 | 0.0 |
| sympy__sympy-13615 | 119 | 82 | 20 | 40 | False | 162 | 106 | 21 | 40 | False | 7 | 0 | 3.1 |
| sympy__sympy-12419 | 192 | 143 | 25 | 40 | False | 157 | 93 | 27 | 40 | False | 10 | 0 | 5.2 |
| pytest-dev__pytest-5262 | 153 | 114 | 20 | 33 | True | 157 | 98 | 28 | 37 | True | 7 | 4 | 4.1 |
| django__django-11133 | 127 | 80 | 19 | 40 | True | 152 | 73 | 29 | 40 | True | 5 | 0 | 2.8 |
| django__django-11265 | 116 | 73 | 23 | 40 | False | 150 | 87 | 23 | 40 | False | 3 | 3 | 1.6 |
| django__django-11206 | 83 | 59 | 8 | 20 | False | 148 | 93 | 18 | 32 | True | 5 | 0 | 2.9 |
| django__django-11211 | 220 | 172 | 22 | 40 | False | 148 | 93 | 27 | 40 | False | 3 | 0 | 1.9 |
| pydata__xarray-2905 | 158 | 94 | 46 | 40 | True | 147 | 84 | 41 | 40 | True | 6 | 1 | 3.0 |
| pylint-dev__pylint-4970 | 188 | 145 | 24 | 40 | False | 146 | 108 | 18 | 40 | False | 5 | 1 | 2.6 |
| pylint-dev__pylint-4604 | 135 | 109 | 16 | 40 | False | 142 | 68 | 36 | 40 | False | 3 | 1 | 1.6 |
| django__django-10097 | 137 | 74 | 37 | 40 | False | 142 | 95 | 22 | 40 | False | 0 | 0 | 0.0 |
| django__django-10914 | 134 | 81 | 29 | 31 | True | 138 | 95 | 16 | 40 | True | 4 | 0 | 2.3 |
| pydata__xarray-3095 | 154 | 107 | 30 | 40 | True | 139 | 87 | 25 | 40 | True | 5 | 1 | 2.9 |
| pydata__xarray-3305 | 143 | 88 | 31 | 40 | False | 138 | 82 | 38 | 40 | True | 0 | 2 | 0.0 |
| django__django-11239 | 144 | 99 | 18 | 32 | False | 132 | 96 | 12 | 34 | False | 4 | 1 | 2.0 |
| psf__requests-1142 | 126 | 102 | 12 | 40 | False | 132 | 89 | 17 | 40 | False | 6 | 2 | 3.3 |
| pylint-dev__pylint-4661 | 122 | 82 | 23 | 32 | False | 131 | 91 | 17 | 35 | False | 4 | 2 | 2.5 |
| django__django-11149 | 139 | 98 | 16 | 38 | False | 128 | 96 | 11 | 30 | True | 3 | 2 | 1.6 |
| psf__requests-1724 | 122 | 53 | 52 | 23 | False | 125 | 55 | 55 | 31 | False | 3 | 1 | 1.8 |
| django__django-11276 | 109 | 69 | 15 | 40 | True | 123 | 64 | 27 | 31 | True | 4 | 1 | 2.4 |
| django__django-11141 | 139 | 87 | 19 | 31 | False | 121 | 83 | 10 | 28 | False | 5 | 1 | 2.3 |
| sympy__sympy-13091 | 103 | 76 | 15 | 36 | False | 121 | 77 | 22 | 30 | False | 3 | 1 | 1.6 |
| django__django-10999 | 135 | 91 | 14 | 27 | False | 116 | 71 | 13 | 26 | False | 3 | 1 | 1.6 |
| django__django-11119 | 166 | 120 | 17 | 40 | True | 110 | 71 | 14 | 26 | True | 3 | 1 | 1.4 |
| sympy__sympy-13031 | 114 | 90 | 13 | 37 | False | 110 | 81 | 14 | 25 | False | 5 | 0 | 2.6 |
| sympy__sympy-12481 | 139 | 102 | 20 | 40 | True | 106 | 81 | 10 | 26 | True | 5 | 0 | 2.9 |
| psf__requests-1766 | 68 | 39 | 19 | 22 | False | 104 | 57 | 28 | 31 | True | 4 | 0 | 1.9 |
| django__django-11066 | 81 | 55 | 9 | 22 | True | 100 | 61 | 16 | 24 | True | 1 | 1 | 0.6 |
| sympy__sympy-12096 | 142 | 65 | 61 | 30 | True | 98 | 70 | 12 | 29 | True | 4 | 1 | 2.1 |
| sympy__sympy-13551 | 144 | 106 | 19 | 40 | False | 98 | 62 | 15 | 35 | False | 6 | 1 | 2.3 |
| pytest-dev__pytest-5631 | 168 | 134 | 15 | 36 | True | 96 | 57 | 17 | 27 | True | 5 | 0 | 2.8 |
| django__django-11099 | 88 | 67 | 7 | 24 | True | 93 | 53 | 18 | 27 | True | 4 | 1 | 2.0 |
| pydata__xarray-3677 | 122 | 63 | 39 | 36 | True | 92 | 48 | 27 | 31 | True | 4 | 0 | 2.2 |
| sympy__sympy-13372 | 54 | 34 | 10 | 20 | True | 85 | 56 | 14 | 22 | False | 4 | 0 | 2.2 |
| sympy__sympy-11618 | 130 | 96 | 17 | 36 | True | 82 | 58 | 12 | 27 | False | 2 | 0 | 1.1 |
| pydata__xarray-3151 | 154 | 111 | 28 | 34 | True | 76 | 47 | 16 | 23 | True | 4 | 1 | 2.2 |
| sphinx-doc__sphinx-10466 | 72 | 47 | 16 | 18 | True | 72 | 41 | 15 | 18 | True | 4 | 0 | 2.5 |
| sympy__sympy-13480 | 60 | 41 | 9 | 19 | True | 67 | 39 | 10 | 17 | True | 4 | 0 | 2.4 |
| pytest-dev__pytest-5809 | 53 | 31 | 13 | 17 | True | 59 | 32 | 14 | 18 | True | 1 | 1 | 0.6 |
| sphinx-doc__sphinx-10449 | 257 | 180 | 47 | 40 | False | 42 | 32 | 5 | 12 | False | 1 | 0 | 0.5 |
