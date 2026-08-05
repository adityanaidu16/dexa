"""Database engine + session. SQLite for dev, Postgres in production — one URL changes it.

    DATABASE_URL=sqlite:///./dexa.db                     # default (dev)
    DATABASE_URL=postgresql+psycopg://user:pw@host/dexa  # production

Nothing else in the control plane knows which one it is.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DEFAULT_URL = "sqlite:///./dexa.db"


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None):
    url = url or os.environ.get("DATABASE_URL", DEFAULT_URL)
    # check_same_thread=False lets the FastAPI thread pool share a SQLite connection safely
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, future=True, **kw)


_engine = None
_Session = None


def init(url: str | None = None):
    """Create the engine + tables. Idempotent; call at startup or in tests."""
    global _engine, _Session
    _engine = make_engine(url)
    from . import models  # noqa: F401  (register tables on Base)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def session():
    if _Session is None:
        init()
    return _Session()


def reset_for_tests(url: str = "sqlite:///:memory:"):
    """Fresh in-memory DB for a test; returns the engine."""
    global _engine, _Session
    _engine = make_engine(url)
    from . import models  # noqa: F401
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine
