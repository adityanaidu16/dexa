"""Persistent, portable session state — the system-layer wedge.

A session's KV cache is serialized to a portable object that can be saved, torn
down, and re-attached in a fresh process / on a different GPU / after a restart
or spot preemption — resuming decode with ~0 re-prefill and identical output.
This is the capability ephemeral per-instance prefix caches structurally cannot
provide. See benchmarks/persist_demo.py.

    from dexa.session import save_kvcache, load_kvcache, SessionStore
"""

from dexa.session.state import save_kvcache, load_kvcache, kvcache_nbytes
from dexa.session.store import SessionStore

__all__ = ["save_kvcache", "load_kvcache", "kvcache_nbytes", "SessionStore"]
