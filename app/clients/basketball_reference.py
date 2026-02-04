from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlencode

from app.clients.http_utils import get_with_retries
from app.core.config import settings

PLAYER_LINK_PATTERN = re.compile(r"/players/[a-z]/[a-z0-9]+\.html")


def _build_headers() -> dict[str, str]:
    return {
        "User-Agent": settings.basketball_reference_user_agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }


def search_player_slug(player_name: str) -> str | None:
    base = settings.basketball_reference_base_url.rstrip("/")
    query = urlencode({"search": player_name})
    url = f"{base}/search/search.fcgi?{query}"
    response = get_with_retries(
        url=url,
        headers=_build_headers(),
        timeout=settings.basketball_reference_timeout_seconds,
        impersonate=settings.nba_stats_impersonate,
        proxy=settings.basketball_reference_proxy,
        max_retries=settings.basketball_reference_max_retries,
        backoff_seconds=settings.basketball_reference_backoff_seconds,
        should_retry=lambda status: status in {403, 429, 500, 502, 503, 504},
    )
    html = response["text"]
    matches = PLAYER_LINK_PATTERN.findall(html)
    if not matches:
        return None
    return matches[0]


def fetch_player_gamelog_html(player_slug: str, season_end_year: int) -> str:
    base = settings.basketball_reference_base_url.rstrip("/")
    slug = player_slug.replace(".html", "")
    url = f"{base}{slug}/gamelog/{season_end_year}"
    response = get_with_retries(
        url=url,
        headers=_build_headers(),
        timeout=settings.basketball_reference_timeout_seconds,
        impersonate=settings.nba_stats_impersonate,
        proxy=settings.basketball_reference_proxy,
        max_retries=settings.basketball_reference_max_retries,
        backoff_seconds=settings.basketball_reference_backoff_seconds,
        should_retry=lambda status: status in {403, 429, 500, 502, 503, 504},
    )
    return response["text"]
