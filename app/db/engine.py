from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import settings


@lru_cache(maxsize=8)
def _engine_for_url(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)


def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or settings.database_url
    if not url:
        raise ValueError("DATABASE_URL is not set")
    return _engine_for_url(url)
