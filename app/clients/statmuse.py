from __future__ import annotations

import re
from urllib.parse import quote

from app.clients.http_utils import get_with_retries
from app.core.config import settings


def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": settings.statmuse_user_agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }


def build_ask_url(query: str) -> str:
    base = settings.statmuse_base_url.rstrip("/")
    slug = quote(query.lower().strip())
    return f"{base}/nba/ask/{slug}"


def fetch_ask_html(query: str) -> str:
    url = build_ask_url(query)
    response = get_with_retries(
        url=url,
        headers=_build_headers(),
        timeout=settings.statmuse_timeout_seconds,
        impersonate=settings.nba_stats_impersonate,
        proxy=settings.statmuse_proxy,
        max_retries=settings.statmuse_max_retries,
        backoff_seconds=settings.statmuse_backoff_seconds,
        should_retry=lambda status: status in {403, 429, 500, 502, 503, 504},
    )
    return response["text"]


def build_player_gamelog_query(player_name: str, season_end_year: int) -> str:
    return f"{player_name} game log {season_end_year}"
