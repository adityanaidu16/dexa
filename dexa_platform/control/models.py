"""Control-plane schema.

Design notes:
  * Screenshots and completions are never stored — only counts and dollars.
  * Credit balance is derived, not mutated per request: granted (CreditLedger) minus spent
    (sum of UsageDaily.dexa_usd). Avoids a hot per-request write to a balance column.
  * Usage is a daily rollup per key, so metering is O(1 upsert) per request instead of a row
    per request.
"""

from __future__ import annotations

import datetime as dt
import secrets

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String,
                        UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("org"))
    name: Mapped[str] = mapped_column(String, default="")
    mode: Mapped[str] = mapped_column(String, default="hosted")        # hosted | byoc
    backend_url: Mapped[str] = mapped_column(String, default="")       # empty => hosted pool
    backend_model: Mapped[str] = mapped_column(String, default="dexa-cua-vlm")
    default_baseline: Mapped[str] = mapped_column(String, default="gpt-4o")
    cache_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    users: Mapped[list["User"]] = relationship(back_populates="org")
    keys: Mapped[list["ApiKey"]] = relationship(back_populates="org")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("provider", "subject", name="uq_identity"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("user"))
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"))
    provider: Mapped[str] = mapped_column(String)                      # github | google | ...
    subject: Mapped[str] = mapped_column(String)                      # provider's user id
    email: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    org: Mapped[Org] = relationship(back_populates="users")


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("key"))
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"), index=True)
    name: Mapped[str] = mapped_column(String, default="default")
    prefix: Mapped[str] = mapped_column(String, index=True)            # visible, e.g. dexa_live_1a2b
    key_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    scopes: Mapped[str] = mapped_column(String, default="inference")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    org: Mapped[Org] = relationship(back_populates="keys")

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class UsageDaily(Base):
    __tablename__ = "usage_daily"
    __table_args__ = (UniqueConstraint("key_id", "day", name="uq_key_day"),)
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("use"))
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"), index=True)
    key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    day: Mapped[str] = mapped_column(String, index=True)              # YYYY-MM-DD (UTC)
    requests: Mapped[int] = mapped_column(Integer, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    dexa_usd: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_usd: Mapped[float] = mapped_column(Float, default=0.0)
    saved_usd: Mapped[float] = mapped_column(Float, default=0.0)
    dexa_image_tokens: Mapped[int] = mapped_column(Integer, default=0)
    baseline_image_tokens: Mapped[int] = mapped_column(Integer, default=0)
    redundant_sum: Mapped[float] = mapped_column(Float, default=0.0)
    redundant_n: Mapped[int] = mapped_column(Integer, default=0)


class CreditLedger(Base):
    __tablename__ = "credit_ledger"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("cr"))
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"), index=True)
    delta_usd: Mapped[float] = mapped_column(Float)                   # +grant / +topup / -adjust
    reason: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
