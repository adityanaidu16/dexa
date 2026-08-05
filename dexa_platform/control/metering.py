"""Durable usage metering + credit accounting.

One upsert per request into a per-key daily rollup, plus a derived credit balance (granted
minus spent). The gateway calls `record()` with plain numbers so the control plane stays
decoupled from the data plane's pricing types. `quota_ok()` is the PLG gate: while free
credit remains, requests flow; when it's exhausted, the gateway returns 402 until the org
adds a payment method (a later phase).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select

from .models import ApiKey, CreditLedger, UsageDaily


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def granted(session, org_id: str) -> float:
    return float(session.execute(
        select(func.coalesce(func.sum(CreditLedger.delta_usd), 0.0))
        .where(CreditLedger.org_id == org_id)).scalar() or 0.0)


def spent(session, org_id: str) -> float:
    return float(session.execute(
        select(func.coalesce(func.sum(UsageDaily.dexa_usd), 0.0))
        .where(UsageDaily.org_id == org_id)).scalar() or 0.0)


def remaining_credit(session, org_id: str) -> float:
    return granted(session, org_id) - spent(session, org_id)


def quota_ok(session, org_id: str) -> bool:
    return remaining_credit(session, org_id) > 0.0


def record(session, org_id: str, key_id: str, *, dexa_usd: float, baseline_usd: float,
           saved_usd: float, dexa_image_tokens: int, baseline_image_tokens: int,
           redundant_frac: float, cache_hit: bool, day: str | None = None) -> None:
    day = day or _today()
    row = session.execute(
        select(UsageDaily).where(UsageDaily.key_id == key_id, UsageDaily.day == day)
    ).scalar_one_or_none()
    if row is None:
        row = UsageDaily(org_id=org_id, key_id=key_id, day=day)
        session.add(row)
        session.flush()          # populate column defaults (0/0.0) before we increment
    row.requests += 1
    row.cache_hits += int(cache_hit)
    row.dexa_usd += dexa_usd
    row.baseline_usd += baseline_usd
    row.saved_usd += saved_usd
    row.dexa_image_tokens += dexa_image_tokens
    row.baseline_image_tokens += baseline_image_tokens
    row.redundant_sum += redundant_frac
    row.redundant_n += 1

    key = session.get(ApiKey, key_id)
    if key is not None:
        key.last_used_at = dt.datetime.now(dt.timezone.utc)
    session.commit()


def _agg(rows) -> dict:
    requests = sum(r.requests for r in rows)
    dexa = sum(r.dexa_usd for r in rows)
    base = sum(r.baseline_usd for r in rows)
    saved = sum(r.saved_usd for r in rows)
    rn = sum(r.redundant_n for r in rows)
    rs = sum(r.redundant_sum for r in rows)
    return {
        "requests": requests,
        "cache_hits": sum(r.cache_hits for r in rows),
        "dexa_usd": round(dexa, 6),
        "baseline_usd": round(base, 6),
        "saved_usd": round(saved, 6),
        "saved_pct": round(100 * saved / base, 2) if base else 0.0,
        "x_cheaper": round(base / dexa, 2) if dexa else 0.0,
        "avg_screen_redundancy_pct": round(100 * rs / rn, 1) if rn else 0.0,
        "dexa_image_tokens": sum(r.dexa_image_tokens for r in rows),
        "baseline_image_tokens": sum(r.baseline_image_tokens for r in rows),
    }


def usage_snapshot(session, org_id: str) -> dict:
    rows = list(session.execute(
        select(UsageDaily).where(UsageDaily.org_id == org_id)).scalars())
    by_key: dict[str, list] = {}
    for r in rows:
        by_key.setdefault(r.key_id, []).append(r)
    return {
        "total": _agg(rows),
        "keys": {kid: _agg(rs) for kid, rs in by_key.items()},
        "credit": {
            "granted_usd": round(granted(session, org_id), 4),
            "spent_usd": round(spent(session, org_id), 6),
            "remaining_usd": round(remaining_credit(session, org_id), 6),
        },
    }


def global_snapshot(session) -> dict:
    """All orgs aggregated + per-org — powers the ops/demo dashboard."""
    from .models import Org
    rows = list(session.execute(select(UsageDaily)).scalars())
    by_org: dict[str, list] = {}
    for r in rows:
        by_org.setdefault(r.org_id, []).append(r)
    names = {o.id: o.name for o in session.execute(select(Org)).scalars()}
    return {
        "total": _agg(rows),
        "sessions": {names.get(oid, oid): _agg(rs) for oid, rs in by_org.items()},
        "session_count": len(by_org),
    }
