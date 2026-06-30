# Integrating Dexa — persistent, portable KV state for your serving stack

Dexa is **not an inference server.** It's the KV-state layer that plugs *into*
the open-source engine you already run (vLLM today, SGLang next), giving your
sessions three properties ephemeral per-instance prefix caching can't:

- **Persistent** — a session's KV survives a process restart.
- **Portable** — resume it on *any* worker (cross-replica, or after a spot
  preemption), with ~0 re-prefill and identical output.
- **Bounded** — compact the cold tail so long sessions stay cheap to hold and
  move (keep recent context raw; see [`docs/CARTRIDGES.md`](CARTRIDGES.md)).

You keep your engine, your OpenAI-compatible API, and your agent harness. Dexa
is infrastructure underneath.

## Try it in 2 minutes (no GPU)

The wedge is reproducible on a laptop with a small model:

```bash
pip install -e '.[hf,bench]'

# resume-from-state vs cold re-prefill (lossless, 14–25× faster, grows with length)
python benchmarks/persist_demo.py bench --model HuggingFaceTB/SmolLM2-360M-Instruct --lengths 256,1024,4096

# the "survives a restart" demo — two separate processes:
python benchmarks/persist_demo.py save   --model HuggingFaceTB/SmolLM2-360M-Instruct --length 2000 --session demo
#   ...kill the process / restart the box...
python benchmarks/persist_demo.py resume --model HuggingFaceTB/SmolLM2-360M-Instruct --session demo
```

## Production: one flag on your vLLM server

Dexa implements vLLM's **KV-connector** interface (the same plug-point LMCache
uses). Enable it on your existing `vllm serve` — no code change to your app:

```bash
pip install 'dexa[vllm]'

vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --kv-transfer-config '{"kv_connector":"DexaConnector",
                         "kv_connector_module_path":"dexa.engine.vllm_connector",
                         "kv_role":"kv_both"}'
```

Now vLLM persists/restores KV through Dexa: an identical context prefix on
another replica — or the same session after a restart — **loads from Dexa
instead of re-prefilling**. Point your OpenAI client at vLLM exactly as before.

> Status: the connector is built against vLLM's documented `KVConnectorBase_V1`
> interface and is **pending validation on a real vLLM** (its method signatures
> are version-specific). The engine-agnostic state lifecycle it drives —
> extract → persist → restore → resume — is CPU-validated (`tests/test_session*`,
> `benchmarks/persist_demo.py`).

## Library mode (custom serving / CPU testing)

If you're not on vLLM, or want to test the lifecycle directly, drive the
engine-agnostic core yourself:

```python
from dexa.engine.hf_backend import HFBackend      # or your ModelBackend
from dexa.serving import SessionManager
from dexa.session.store import SessionStore

mgr = SessionManager(HFBackend("Qwen/Qwen2.5-Coder-7B-Instruct", device="cuda"),
                     store=SessionStore("/mnt/sessions"))

# each turn prefills only the NEW turn against the persisted session, not the
# whole history; the session restores on any worker after a restart.
text, info = mgr.turn("session-42", "add a test for parse_config()")
print(text, f"(prefill saved {info.prefill_savings:.0%} this turn)")
```

`SessionStore` is directory-backed by default; back it with object storage / a
shared NVMe tier in production (the interface is the same).

## How it works

1. **Extract** — after prefill, Dexa captures the engine's KV for a session.
2. **Persist** — the KV (raw, lossless; or compacted for the cold tail) is
   written to the store, keyed by content hash / session id.
3. **Restore** — on a request whose prefix matches a persisted session — on any
   worker, after any restart — Dexa loads the KV and the engine skips re-prefill.
4. **Resume** — decode continues with identical output; only the new delta is
   prefilled.

The economic upshot: because the state is portable and survives preemption, the
session can run on the cheapest interruptible/spot capacity — which stateless,
per-instance caches cannot.
