"""Shared singleton instances for cache and logging setup."""
from __future__ import annotations

from app.clients.cache import ResponseCache

_cache: ResponseCache | None = None


def get_shared_cache() -> ResponseCache:
    global _cache  # noqa: PLW0603
    if _cache is None:
        from app.core.config import settings
        _cache = ResponseCache(
            cache_dir=settings.cache_dir,
            default_ttl_seconds=settings.cache_default_ttl_seconds,
        )
        _cache.set_ttl("nba_stats", settings.cache_nba_stats_ttl_seconds)
        _cache.set_ttl("prizepicks", settings.cache_prizepicks_ttl_seconds)
        _cache.set_ttl("basketball_reference", settings.cache_bref_ttl_seconds)
        _cache.set_ttl("statmuse", settings.cache_statmuse_ttl_seconds)
    return _cache
