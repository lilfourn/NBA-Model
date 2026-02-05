from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import settings


@lru_cache(maxsize=8)
def _engine_for_url(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)


def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or settings.database_url
    if not url:
        raise ValueError("DATABASE_URL is not set")
    return _engine_for_url(url)


@lru_cache(maxsize=8)
def _async_engine_for_url(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def get_async_engine(database_url: str | None = None) -> AsyncEngine:
    url = database_url or settings.database_url
    if not url:
        raise ValueError("DATABASE_URL is not set")
    return _async_engine_for_url(url)
