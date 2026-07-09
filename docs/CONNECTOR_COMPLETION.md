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

## Status after the end-to-end run (vLLM 0.24.0, OPT-125m, A10G)

`scripts/modal_connector_serve.py` (two identical requests, prefix caching off) got
the connector to **construct and run inside real vLLM with generation working and
output identical** — so the 3-arg ctor, `_spec`, `_layer_kv_tensors`, and the load
path are correctly wired and nothing raises. **But no KV was saved** (`saved=False`,
no `[dexa] saved KV` log), so cross-request reuse does not yet happen.

**Root cause — save decided one lifecycle phase too late.** The current design queues
saves in `request_finished`, which vLLM calls *after* a request's final forward pass.
But `save_kv_layer` (captures KV) fires *during* forward passes, and
`build_connector_meta` (packs the save plan the worker binds) runs *before* them. So
when `request_finished` marks a request, there is no remaining forward pass and
`save_kv_layer` never captures it.

**Fix (next task) — mirror vLLM's `SharedStorageConnector`:** decide saves in
`build_connector_meta` from `scheduler_output` — mark a request for save on the step
its prefill completes (all prompt tokens computed, not yet in the store) so
`save_kv_layer` captures its KV that same step. The `scheduler_output` structure is
already captured by the probe; `request_finished` can drop back to just freeing
blocks / returning `(False, None)`. Then re-run `modal_connector_serve.py`: request 1
should log `[dexa] saved KV`, request 2 `[dexa] store HIT`, with a KV file on disk.

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
