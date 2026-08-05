"""Accounts + key lifecycle — the PLG onboarding path.

`signup_oauth` is the whole activation: an OAuth identity comes in, and an org, a user, a
first API key, and a free-credit grant come out — no card, idempotent on repeat sign-in. The
rest is key management the console calls: create, list, revoke, rotate.
"""

from __future__ import annotations

import datetime as dt
import os

from sqlalchemy import select

from . import keys as keymod
from .models import ApiKey, CreditLedger, Org, User

FREE_CREDIT_USD = float(os.environ.get("DEXA_FREE_CREDIT_USD", "5.0"))


class NewKey:
    """A freshly minted key — carries the plaintext secret ONCE, for the caller to show."""
    def __init__(self, record: ApiKey, secret: str):
        self.record = record
        self.secret = secret  # never persisted; display once


def _mint_key(session, org_id: str, name: str, env: str) -> NewKey:
    secret = keymod.generate_secret(env)
    rec = ApiKey(org_id=org_id, name=name, prefix=keymod.visible_prefix(secret),
                 key_hash=keymod.hash_secret(secret))
    session.add(rec)
    session.flush()
    return NewKey(rec, secret)


def signup_oauth(session, provider: str, subject: str, email: str = "",
                 org_name: str = "") -> tuple[User, NewKey | None]:
    """Get-or-create for an OAuth identity. Returns (user, new_key). new_key is non-None only
    on first sign-up (the one time we can show a secret)."""
    existing = session.execute(
        select(User).where(User.provider == provider, User.subject == subject)
    ).scalar_one_or_none()
    if existing:
        return existing, None

    org = Org(name=org_name or (email.split("@")[0] if email else "workspace"))
    session.add(org)
    session.flush()
    session.add(CreditLedger(org_id=org.id, delta_usd=FREE_CREDIT_USD, reason="signup_grant"))
    user = User(org_id=org.id, provider=provider, subject=subject, email=email)
    session.add(user)
    new_key = _mint_key(session, org.id, name="default", env="live")
    session.commit()
    return user, new_key


def create_key(session, org_id: str, name: str = "key", env: str = "live") -> NewKey:
    nk = _mint_key(session, org_id, name, env)
    session.commit()
    return nk


def list_keys(session, org_id: str) -> list[ApiKey]:
    return list(session.execute(
        select(ApiKey).where(ApiKey.org_id == org_id).order_by(ApiKey.created_at)
    ).scalars())


def revoke_key(session, org_id: str, key_id: str) -> bool:
    rec = session.get(ApiKey, key_id)
    if not rec or rec.org_id != org_id or rec.revoked_at is not None:
        return False
    rec.revoked_at = dt.datetime.now(dt.timezone.utc)
    session.commit()
    return True


def rotate_key(session, org_id: str, key_id: str) -> NewKey | None:
    """Revoke an existing key and mint a replacement with the same name."""
    rec = session.get(ApiKey, key_id)
    if not rec or rec.org_id != org_id:
        return None
    rec.revoked_at = dt.datetime.now(dt.timezone.utc)
    nk = _mint_key(session, org_id, name=rec.name, env="live")
    session.commit()
    return nk


def get_org(session, org_id: str) -> Org | None:
    return session.get(Org, org_id)
