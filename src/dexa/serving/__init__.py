"""Engine-agnostic stateful-session core.

Dexa is **not** an inference server — it's the KV-state layer that plugs into
open-source serving engines (vLLM via its KV-connector, SGLang next). This
package holds the engine-agnostic lifecycle the integrations call into:

    create a session -> extend it with delta turns (prefill only the new tokens)
    -> persist between turns -> restore on any worker after a restart.

It runs against any :class:`~dexa.engine.base.ModelBackend` (HF backend for
CPU-validated reference + library use); the vLLM connector
(:mod:`dexa.engine.vllm_connector`) drives the same lifecycle over vLLM's paged
KV so a builder keeps their vLLM server + OpenAI API unchanged.

    from dexa.serving import SessionManager
"""

from dexa.serving.session_manager import SessionManager, SessionInfo

__all__ = ["SessionManager", "SessionInfo"]
