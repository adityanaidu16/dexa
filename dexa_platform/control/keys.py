"""API key material — generation, hashing, verification.

A key looks like `dexa_live_<random>` (or `dexa_test_<random>`). The full secret is shown to
the user exactly once, at creation. We persist only its SHA-256 hash and a short visible
prefix (for the console's "dexa_live_9f2c…" display and for support). A presented key is
verified by hashing it and matching the stored hash — the plaintext is never stored or logged.
"""

from __future__ import annotations

import hashlib
import secrets

PREFIX_VISIBLE = 14  # chars shown in the console, e.g. "dexa_live_9f2c"


def generate_secret(env: str = "live") -> str:
    if env not in ("live", "test"):
        raise ValueError("env must be 'live' or 'test'")
    return f"dexa_{env}_{secrets.token_urlsafe(24)}"


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def visible_prefix(secret: str) -> str:
    return secret[:PREFIX_VISIBLE]


def looks_like_key(s: str | None) -> bool:
    return bool(s) and s.startswith(("dexa_live_", "dexa_test_"))
