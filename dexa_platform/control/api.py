"""Control-plane API — what the console (and a CLI) call.

PLG sign-up is public: an OAuth identity in, an account + first key out. Management endpoints
(keys, usage) authenticate with a Bearer API key that resolves to its org. In production the
console would carry a user session (a JWT minted from the OAuth callback) distinct from
inference keys; key-auth here keeps the MVP self-contained and is called out as such.

    uvicorn dexa_platform.control.api:app --port 8090
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from . import accounts, db, keys as keymod, metering
from .models import ApiKey

app = FastAPI(title="Dexa Control Plane", version="0.1.0")


@app.on_event("startup")
def _startup():
    db.init()


def org_from_key(authorization: str = Header(default="")) -> str:
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if not keymod.looks_like_key(token):
        raise HTTPException(401, "missing or malformed API key")
    with db.session() as s:
        rec = s.execute(select(ApiKey).where(
            ApiKey.key_hash == keymod.hash_secret(token))).scalar_one_or_none()
        if rec is None or rec.revoked_at is not None:
            raise HTTPException(401, "invalid or revoked API key")
        return rec.org_id


class SignupIn(BaseModel):
    provider: str = "github"
    subject: str
    email: str = ""
    org_name: str = ""


class KeyIn(BaseModel):
    name: str = "key"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/signup")
def signup(body: SignupIn):
    """Public PLG onboarding. In prod, call this from your OAuth callback with a verified
    identity. Returns the first key secret ONCE — it is never retrievable again."""
    with db.session() as s:
        user, new_key = accounts.signup_oauth(
            s, body.provider, body.subject, body.email, body.org_name)
        org_id = user.org_id
        resp = {"org_id": org_id, "user_id": user.id, "returning": new_key is None}
        if new_key is not None:
            resp["api_key"] = new_key.secret            # show once
            resp["key_prefix"] = new_key.record.prefix
        resp["credit"] = metering.usage_snapshot(s, org_id)["credit"]
        return resp


@app.get("/v1/me")
def me(org_id: str = Depends(org_from_key)):
    with db.session() as s:
        org = accounts.get_org(s, org_id)
        snap = metering.usage_snapshot(s, org_id)
        return {"org": {"id": org.id, "name": org.name, "mode": org.mode,
                        "backend_model": org.backend_model, "cache_enabled": org.cache_enabled},
                "credit": snap["credit"], "usage_total": snap["total"]}


@app.get("/v1/keys")
def get_keys(org_id: str = Depends(org_from_key)):
    with db.session() as s:
        return {"keys": [
            {"id": k.id, "name": k.name, "prefix": k.prefix, "scopes": k.scopes,
             "active": k.active,
             "created_at": k.created_at.isoformat() if k.created_at else None,
             "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None}
            for k in accounts.list_keys(s, org_id)]}


@app.post("/v1/keys")
def post_key(body: KeyIn, org_id: str = Depends(org_from_key)):
    with db.session() as s:
        nk = accounts.create_key(s, org_id, name=body.name)
        return {"api_key": nk.secret, "id": nk.record.id, "prefix": nk.record.prefix}


@app.post("/v1/keys/{key_id}/rotate")
def rotate(key_id: str, org_id: str = Depends(org_from_key)):
    with db.session() as s:
        nk = accounts.rotate_key(s, org_id, key_id)
        if nk is None:
            raise HTTPException(404, "key not found")
        return {"api_key": nk.secret, "id": nk.record.id, "prefix": nk.record.prefix}


@app.delete("/v1/keys/{key_id}")
def revoke(key_id: str, org_id: str = Depends(org_from_key)):
    with db.session() as s:
        if not accounts.revoke_key(s, org_id, key_id):
            raise HTTPException(404, "key not found or already revoked")
        return {"revoked": key_id}


@app.get("/v1/usage")
def usage(org_id: str = Depends(org_from_key)):
    with db.session() as s:
        return metering.usage_snapshot(s, org_id)
