from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import settings


def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or settings.database_url
    if not url:
        raise ValueError("DATABASE_URL is not set")
    return create_engine(url, pool_pre_ping=True, future=True)
