# Paired analysis of `20260915T044016Z-harness-7`

Tasks: 50 off, 50 harness. Errored records: 9.

Errors (excluded from paired statistics; counted as unresolved):

- django__django-11087: AttributeError("'str' object has no attribute 'usage'")
- django__django-11066: AttributeError("'str' object has no attribute 'usage'")
- django__django-10973: AttributeError("'str' object has no attribute 'usage'")
- django__django-10999: AttributeError("'str' object has no attribute 'usage'")
- django__django-10914: AttributeError("'str' object has no attribute 'usage'")
- django__django-10097: AttributeError("'str' object has no attribute 'usage'")
- django__django-11095: AttributeError("'str' object has no attribute 'usage'")
- django__django-10880: AttributeError("'str' object has no attribute 'usage'")
- django__django-11141: AttributeError("'str' object has no attribute 'usage'")

## Paired tasks (n=41)

| metric | off | harness |
|---|---|---|
| resolved (official grade) | 24 | 22 |
| agent loop minus sandbox boot, median (s) | 115 | 143 |
| agent loop minus sandbox boot, mean (s) | 139 | 203 |
| sandbox boot, mean (s) | 43.7 | 2.3 |
| model time, median (s) | 70 | 102 |
| tool time, median (s) | 19 | 16 |
| steps, mean | 33.0 | 31.9 |

Per-task ratio harness/off of the boot-corrected loop: median 1.160, IQR 1.00 to 1.37.
Tasks per GPU-hour on the paired tasks at concurrency 8, boot excluded: off 207, harness 142, ratio 0.684.
Dropping the 3 slowest tasks in each arm: ratio 0.774.

## Speculation accounting (harness arm, paired tasks)

- launches 228 (5.6 per task), hits 162, misses 50, hit rate 0.76
- rule A: 19 hits, 45 misses, hit rate 0.30
- rule B: 143 hits, 5 misses, hit rate 0.97
- saved per hit: median 0.55 s, mean 0.57 s, max 2.3 s; speculative command duration median 0.90 s
- saved per task 2.26 s; killed-run time per task 5.86 s (background, not wall clock); launch/poll/kill round trips are extra sandbox execs of roughly 0.3 to 0.5 s each, not itemized
- ceiling: if every launch had hit with zero overhead, saving would be at most 10.9 s per task against a median loop of 115 s

## Model latency by context depth (both arms pooled)

| steps | calls | median (s) | p90 (s) | max (s) |
|---|---|---|---|---|
| 0 to 9 | 813 | 0.95 | 5.39 | 523 |
| 10 to 19 | 784 | 1.01 | 6.98 | 265 |
| 20 to 29 | 667 | 1.18 | 7.09 | 262 |
| 30 to 39 | 396 | 1.30 | 6.83 | 20 |

Stalls (single model calls over 120 s): 14, total 3162 s, across 12 task runs: django__django-11141 (off, step 24, 260 s), django__django-11133 (off, step 17, 260 s), django__django-11133 (off, step 29, 262 s), sympy__sympy-13480 (off, step 4, 258 s), django__django-11206 (harness, step 0, 168 s), django__django-11149 (harness, step 0, 168 s), django__django-11179 (harness, step 0, 167 s), django__django-11099 (harness, step 0, 168 s), django__django-11133 (harness, step 0, 168 s), django__django-11211 (harness, step 0, 164 s), django__django-11119 (harness, step 0, 168 s), django__django-11163 (harness, step 0, 165 s), django__django-11163 (harness, step 5, 523 s), pytest-dev__pytest-5262 (harness, step 19, 265 s)

Off arm token totals: 17.9M prompt, 304k completion, ratio 59:1.

## Per-task table

| instance | off loop (s) | off model | off tool | off steps | off resolved | harness loop (s) | model | tool | steps | resolved | hits | misses | saved (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pytest-dev__pytest-10081 | 373 | 43 | 318 | 29 | True | 1369 | 102 | 1230 | 40 | False | 2 | 7 | 1.2 |
| django__django-11163 | 88 | 61 | 11 | 31 | True | 881 | 764 | 65 | 27 | True | 3 | 1 | 1.6 |
| pytest-dev__pytest-5262 | 134 | 88 | 23 | 40 | True | 383 | 344 | 18 | 34 | True | 8 | 0 | 4.6 |
| django__django-11119 | 124 | 83 | 18 | 31 | True | 341 | 294 | 15 | 40 | True | 7 | 0 | 4.2 |
| django__django-11211 | 98 | 68 | 14 | 35 | False | 316 | 268 | 19 | 40 | False | 6 | 1 | 3.5 |
| django__django-11133 | 611 | 580 | 12 | 30 | True | 310 | 267 | 13 | 40 | True | 7 | 1 | 3.7 |
| django__django-11099 | 76 | 40 | 19 | 25 | True | 285 | 251 | 11 | 31 | True | 4 | 1 | 2.3 |
| django__django-11179 | 98 | 58 | 18 | 33 | True | 273 | 241 | 14 | 27 | True | 2 | 1 | 1.1 |
| django__django-11149 | 166 | 102 | 25 | 40 | False | 271 | 245 | 9 | 25 | False | 2 | 2 | 1.2 |
| django__django-11206 | 107 | 74 | 15 | 30 | False | 245 | 222 | 8 | 23 | True | 3 | 0 | 1.6 |
| sphinx-doc__sphinx-10323 | 198 | 159 | 19 | 40 | False | 237 | 184 | 24 | 40 | False | 6 | 6 | 3.4 |
| sphinx-doc__sphinx-10449 | 152 | 112 | 24 | 40 | True | 192 | 148 | 21 | 40 | False | 2 | 5 | 1.2 |
| pylint-dev__pylint-4970 | 145 | 96 | 29 | 40 | False | 177 | 122 | 22 | 40 | False | 5 | 2 | 2.9 |
| pydata__xarray-2905 | 145 | 76 | 47 | 40 | True | 174 | 102 | 48 | 40 | True | 7 | 1 | 4.1 |
| sympy__sympy-12419 | 158 | 105 | 24 | 40 | False | 174 | 121 | 21 | 40 | False | 9 | 0 | 5.0 |
| sphinx-doc__sphinx-10435 | 130 | 86 | 26 | 35 | False | 160 | 120 | 24 | 39 | False | 4 | 1 | 2.5 |
| sympy__sympy-13551 | 135 | 96 | 19 | 40 | False | 156 | 112 | 14 | 40 | False | 10 | 3 | 4.7 |
| django__django-11239 | 133 | 94 | 15 | 35 | True | 151 | 110 | 13 | 34 | False | 2 | 2 | 1.2 |
| pylint-dev__pylint-4661 | 111 | 74 | 21 | 38 | False | 151 | 111 | 18 | 40 | False | 5 | 1 | 2.7 |
| django__django-11265 | 137 | 82 | 30 | 40 | True | 147 | 108 | 17 | 40 | False | 3 | 1 | 1.7 |
| pydata__xarray-3151 | 222 | 118 | 71 | 40 | True | 143 | 88 | 33 | 34 | True | 5 | 3 | 2.7 |
| sympy__sympy-13615 | 154 | 119 | 20 | 37 | False | 142 | 89 | 22 | 40 | False | 5 | 1 | 2.5 |
| pytest-dev__pytest-5631 | 88 | 59 | 18 | 32 | True | 140 | 104 | 17 | 37 | True | 4 | 1 | 2.1 |
| pydata__xarray-3095 | 143 | 67 | 53 | 37 | True | 137 | 82 | 36 | 36 | True | 4 | 1 | 2.3 |
| pydata__xarray-3305 | 103 | 68 | 21 | 40 | False | 129 | 79 | 34 | 40 | False | 2 | 0 | 2.8 |
| pylint-dev__pylint-4604 | 91 | 57 | 21 | 40 | False | 124 | 90 | 20 | 40 | False | 1 | 2 | 0.8 |
| psf__requests-1142 | 102 | 70 | 16 | 37 | False | 118 | 92 | 11 | 36 | False | 4 | 1 | 2.3 |
| pydata__xarray-3677 | 115 | 46 | 48 | 31 | True | 114 | 58 | 40 | 35 | True | 4 | 0 | 2.5 |
| sympy__sympy-12481 | 117 | 85 | 17 | 35 | True | 110 | 86 | 13 | 33 | True | 2 | 1 | 1.2 |
| django__django-11276 | 104 | 65 | 14 | 40 | True | 104 | 62 | 17 | 30 | True | 5 | 0 | 2.9 |
| sympy__sympy-13091 | 78 | 44 | 19 | 29 | False | 86 | 64 | 10 | 31 | False | 3 | 1 | 1.5 |
| sympy__sympy-11618 | 87 | 66 | 12 | 27 | True | 85 | 58 | 11 | 27 | True | 4 | 0 | 2.3 |
| sympy__sympy-12096 | 244 | 89 | 125 | 39 | True | 80 | 53 | 13 | 30 | True | 4 | 0 | 2.3 |
| sphinx-doc__sphinx-10466 | 57 | 33 | 15 | 17 | True | 76 | 54 | 10 | 20 | True | 3 | 1 | 1.6 |
| psf__requests-1724 | 67 | 43 | 14 | 29 | True | 74 | 49 | 16 | 23 | True | 3 | 0 | 1.9 |
| psf__requests-1766 | 51 | 34 | 9 | 18 | True | 62 | 40 | 10 | 21 | True | 3 | 0 | 1.6 |
| pytest-dev__pytest-10051 | 54 | 35 | 10 | 27 | False | 59 | 39 | 9 | 24 | False | 3 | 0 | 1.5 |
| pytest-dev__pytest-5809 | 51 | 31 | 13 | 17 | True | 58 | 31 | 15 | 17 | True | 2 | 2 | 1.2 |
| sympy__sympy-13372 | 88 | 54 | 19 | 28 | False | 58 | 39 | 10 | 19 | True | 3 | 0 | 1.5 |
| sympy__sympy-13480 | 304 | 277 | 19 | 14 | True | 36 | 26 | 6 | 12 | True | 0 | 0 | 0.0 |
| sympy__sympy-13031 | 67 | 47 | 11 | 26 | False | 11 | 7 | 1 | 3 | False | 1 | 0 | 0.5 |
| django__django-10880 | 139 | 98 | 17 | 40 | False | error | | | | | | | |
| django__django-11095 | 103 | 69 | 15 | 34 | True | error | | | | | | | |
| django__django-10097 | 122 | 74 | 27 | 40 | False | error | | | | | | | |
| django__django-10914 | 147 | 103 | 17 | 35 | True | error | | | | | | | |
| django__django-10999 | 184 | 80 | 39 | 31 | False | error | | | | | | | |
| django__django-10973 | 142 | 100 | 18 | 36 | True | error | | | | | | | |
| django__django-11066 | 63 | 40 | 12 | 20 | True | error | | | | | | | |
| django__django-11087 | 123 | 62 | 26 | 24 | False | error | | | | | | | |
| django__django-11141 | 353 | 320 | 13 | 27 | False | error | | | | | | | |
