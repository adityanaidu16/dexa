# vLLM connector completion — from "signatures match" to "moves KV through real vLLM"

**Status.** `src/dexa/engine/vllm_connector.py` (`DexaConnector`) implements the full
vLLM V1 `KVConnectorBase_V1` scheduler/worker lifecycle, and its signatures were
validated against a real vLLM 0.24.0 (10/10 hooks match, zero drift — see
`docs/RESULTS.md`, 2026-07-09). The pure-numpy paged-block ↔ KVCache helpers are
unit-tested everywhere. **But no real request's KV has yet round-tripped through a
live vLLM:** five version-pinned *site shims* deliberately `raise`, and everything
validated end-to-end so far is on `HFBackend`, which is not a serving engine.

**This is the #1 blocker to being useful to ML engineers** — they serve with
vLLM/SGLang, not a HuggingFace forward loop. Closing it turns the "one flag in
vLLM" pitch from unproven into demoable.

## Done-when

Two vLLM instances behind Dexa: instance A serves a long context; instance B
(fresh, or after A is killed) resumes it with **zero re-prefill and correct
output**, driven by a *real request* — not a unit test.

## Pin first

Do **not** chase vLLM's moving V1 API. Pin **one** version in the `[vllm]` extra
(0.24.0, already signature-validated, or the latest stable you'll deploy on) and
target that exact `KVConnectorBase_V1`. Use **LMCache's connector as the reference
implementation** — it solves each shim against the same seam, so this is porting
known-good logic, not inventing it. The Mooncake connectors (`MooncakeConnector`,
`MooncakeStoreConnector`) are a second reference on the identical surface.

## The five site-shims (all currently `raise`)

Implement in dependency order:

| Shim | Must return | Where to get it |
|---|---|---|
| `_spec()` | `ModelSpec(n_layers, n_kv_heads, head_dim, hidden_size, dtype)` | Already solved in `VLLMBackend.__init__` — lift that field-probing verbatim |
| `_layer_kv_tensors(name)` | `(key_cache, value_cache)` views of the registered paged tensor | Attention-backend-specific: FlashAttention packs `[2, num_blocks, block_size, n_kv, d]`; FlashInfer/MLA differ. Pin one backend, handle that layout |
| `_split_kv_layer(kv_layer)` | same split for the tensor handed to `save_kv_layer` | Same layout knowledge |
| `_block_ids(blocks)` | `list[int]` physical block numbers | vLLM V1 `KVCacheBlocks`/block-table object — the shape LMCache reads |
| `_save_geometry(req_id, meta)` | `(T, positions, token_ids)` for the finished request | Carry `token_ids` in the metadata already built in `build_connector_meta` |

## The three real correctness risks

The pure-numpy tests give false confidence here — these only surface against a live
engine:

1. **Partial final block.** A sequence rarely fills its last block; the padding drop
   must be exact or the tail corrupts. `paged_blocks_to_kvcache` handles it in numpy,
   but the *engine* block boundaries must line up.
2. **Tensor parallelism (TP>1).** Each worker holds a *shard* of the KV heads.
   Save/load must be per-shard-consistent, or reuse across a differently-sharded
   instance silently corrupts. This is the single most likely thing to break "resume
   on any instance," and the long pole.
3. **Cross-attention-backend / cross-arch identity.** KV computed with FlashAttention
   on A100 vs FlashInfer on H100 may not be reload-compatible. **Scope the v1 claim
   to same attention-backend + same TP degree** and say so; broaden later with
   measurement, not assumption.

## Deliverable that proves it

Extend `benchmarks/vllm_connector_check.py` tier 2 (`--serve`) into a **two-process
reuse test**: process A serves a prompt (KV saved to a shared dir); process B starts
fresh, sends the same prompt, and we assert `get_num_new_matched_tokens > 0` and
identical greedy output with re-prefill skipped. That single green test is the entire
"one flag in vLLM" promise made real, and it doubles as the lossless-correctness gate
the independent serving benchmarks do **not** provide (see `docs/BENCHMARK_PLAN.md`).

## Discovered ground truth (vLLM 0.24.0, from `scripts/modal_connector_probe.py`)

A live probe run (OPT-125m, A10G) captured the real object shapes each shim needs.
The connector construction and all lifecycle hooks fired successfully; recorded:

**Constructor (blocks loading entirely).** vLLM ≥0.24 validates the ctor at config
time and *rejects* the documented 2-arg form:
`__init__(self, vllm_config, role, kv_cache_config)` must pass all three to
`super().__init__()`. `kv_cache_config` is the source for `_spec` (below).

**`_spec()`** ← `kv_cache_config.kv_cache_groups[0]`
(`vllm.v1.kv_cache_interface.KVCacheGroupSpec`):
- `.kv_cache_spec` (`FullAttentionSpec`): `block_size=16`, `head_size=64`,
  `num_kv_heads=12`, `dtype.itemsize=2` (fp16), `page_size_bytes=49152`
  (= 2·block_size·num_kv_heads·head_size·2 bytes ✓).
- `.layer_names` (len 12) → `n_layers = len(layer_names)`.
- `n_q_heads` / `hidden_size` from `vllm_config.model_config`.

**`_layer_kv_tensors(name)` / `_split_kv_layer(kv_layer)`** ← `register_kv_caches`
hands a dict `{layer_name: tensor}` (12 entries, keys
`"model.decoder.layers.{i}.self_attn.attn"`); each tensor is
**`[num_blocks, 2, block_size, n_kv_heads, head_dim]`** (`[23151, 2, 16, 12, 64]`,
fp16, cuda). Dim 1 is the K/V pair → `key = t[:, 0]`, `value = t[:, 1]`, each
`[num_blocks, block_size, n_kv_heads, head_dim]`. (Verify `save_kv_layer`'s
`kv_layer` matches this layout — its dump was truncated; it is the same registered
tensor, so almost certainly identical.)

**`_block_ids(blocks)`** ← `update_state_after_alloc`'s `blocks` is
`vllm.v1.core.kv_cache_manager.KVCacheBlocks` with `.blocks` = a tuple (one entry
per KV-cache group) of lists of `vllm.v1.core.kv_cache_utils.KVCacheBlock`. Each
`KVCacheBlock` carries `.block_id` → `[b.block_id for b in blocks.blocks[0]]`
(single group here; flatten groups if >1).

**`_save_geometry(req_id, meta)`** ← the `request`
(`vllm.v1.request.Request`) has `prompt_token_ids` (plain list) and `all_token_ids`
(a `vllm.v1.utils.ConstantList` = prompt+output), plus `request_id`,
`num_prompt_tokens`, `num_tokens`. The worker has no request object at save time, so
**carry the token ids through `build_connector_meta`** (capture them in
`request_finished`); `_save_geometry` then returns `(T, arange(T), token_ids)`.

**Runtime image.** A real vLLM engine probes `nvcc` during KV-cache init, so Modal
runs need a CUDA *devel* base (not runtime-only pip) — see
`scripts/modal_connector_probe.py`.

## Status: WORKS end-to-end (cross-request, TP=1) — validated 2026-07-09

`scripts/modal_connector_serve.py` (vLLM 0.24.0, OPT-125m, A10G, prefix caching off)
now shows full cross-request KV reuse: request 1 logs `[dexa] saved KV: T=43` and
writes a `.npz`; request 2 logs `[dexa] store HIT: 32 external tokens`, loads 2
blocks, re-prefills only the remaining 11 tokens, and returns **bit-identical**
output (`saved=True  identical_output=True`, no crash). See `docs/RESULTS.md`.

Getting here retired, in order (each via one probe/serve Modal run):
1. **ctor** — vLLM ≥0.24 requires 3-arg `(vllm_config, role, kv_cache_config)`.
2. **image** — a real engine needs `nvcc`; use a CUDA *devel* base.
3. **object shapes** — the five shims, implemented against the probe dump (above).
4. **save timing** — decide saves in `build_connector_meta` from
   `scheduler_output.scheduled_new_reqs` (prefill-complete this step), not in
   `request_finished` (fires after the last forward → never captured).
5. **scheduler constraint** — `get_num_new_matched_tokens` must leave ≥1 token and
   claim only whole blocks (vLLM asserts `num_new_tokens > 0`; reuse is
   block-granular). A sub-block prompt yields 0.

**Cross-instance: VALIDATED** (`scripts/modal_connector_xinstance.py`, 2026-07-09).
Two separate Modal containers (distinct vLLM processes, pids 36 vs 216) sharing a
persistent Volume as the store: A saved its KV; B — a fresh process — saw A's KV on
entry, hit the store, loaded it, and returned identical output
(`A_saved=True B_saw_stored_KV=True identical_output=True`). The portable-across-
instances property per-instance prefix caching can't provide.

**Adaptive load-vs-recompute (v1 done).** vLLM's prefill is fast (8B/A100/8k =
617 ms, faster than a 1.7 GB KV load), so blindly loading can be *slower* than
recomputing at short contexts. `get_num_new_matched_tokens` now gates the load on
`load_decision(n_tokens, policy, min_load_tokens, contention_factor)` (pure,
unit-tested): `adaptive` (default) loads only past a calibrated crossover
(`dexa_min_load_tokens`, default 32768, model/hardware/tier specific), so Dexa is
**never worse than re-prefill** — below the crossover it reports 0 and vLLM prefills.
`always`/`never` force the choice. **Contention is now dynamic** (`dexa_contention_aware`,
default on): each step, `build_connector_meta` updates a busyness EMA from the
scheduler batch — `max(scheduled_tokens/max_batch_tokens, running_reqs/max_seqs)` —
and `contention_factor_from_busy` lowers the crossover toward `dexa_contention_floor`
(default 0.1) as the GPU saturates. So under real serving load, re-prefill queues and
competes for compute while a KV load uses idle I/O, and Dexa loads at *short* contexts
where it would otherwise re-prefill — the win single-request benchmarks miss. All
pure decision logic (`load_decision`, `contention_factor_from_busy`) is unit-tested.
**Next:** validate the contention win with a *concurrent* resume benchmark (high
QPS → busy GPU → Dexa loads short contexts and beats queued re-prefill on P99 TTFT),
and an online cost model (fit the a·n + b·n² prefill curve to set the crossover
automatically instead of the calibrated default).

**Remaining (honest):**
- **TP>1** — KV-head sharding; the hard one (save/load must be per-shard-consistent).
- **Chunked prefill** — a large prompt completes over multiple steps and moves to
  `scheduled_cached_reqs`; the save loop only scans `scheduled_new_reqs`, so it
  currently saves only single-step-prefill prompts. Extend to cached reqs.
- **Cross-attention-backend / cross-GPU-arch** portability — scope the claim to
  same-backend + same-TP until measured.

## Sequencing

`_spec` → the two tensor-split shims for one pinned attention backend (TP=1) →
`_block_ids` / `_save_geometry` → the two-process reuse test → **then** TP>1. Get
TP=1 cross-instance reuse working and demoable before touching sharding.

**Rough effort:** the shims are ~1–2 focused days against a pinned version with
LMCache open beside you; TP>1 correctness is the long pole (a few more days +
multi-GPU testing). Bounded work — the hard design (lifecycle, block↔numpy,
persistence) is done and tested.

## Why this must land before benchmarking

Every independent serving harness (`vllm bench serve`, AIPerf + Mooncake) drives a
*working* vLLM server. None of the benchmarking in `docs/BENCHMARK_PLAN.md` can run
until the connector completes. This is the strict prerequisite for the whole
"prove it to ML engineers" path.
