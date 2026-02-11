from __future__ import annotations

import re
from urllib.parse import urlencode

from app.clients.base import CrawlerClient
from app.clients.shared import get_shared_cache
from app.core.config import settings

PLAYER_LINK_PATTERN = re.compile(r"/players/[a-z]/[a-z0-9]+\.html")

_client: CrawlerClient | None = None


def _get_client() -> CrawlerClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = CrawlerClient(
            source_name="basketball_reference",
            max_retries=settings.basketball_reference_max_retries,
            backoff_seconds=settings.basketball_reference_backoff_seconds,
            read_timeout=float(settings.basketball_reference_timeout_seconds),
            impersonate=settings.nba_stats_impersonate,
            proxy=settings.basketball_reference_proxy or None,
            default_headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": settings.basketball_reference_base_url + "/",
            },
            should_retry=lambda status: status in {403, 429, 500, 502, 503, 504},
            min_request_interval=2.0,
            cache=get_shared_cache(),
        )
    return _client


def search_player_slug(player_name: str) -> str | None:
    base = settings.basketball_reference_base_url.rstrip("/")
    query = urlencode({"search": player_name})
    url = f"{base}/search/search.fcgi?{query}"
    client = _get_client()
    result = client.get(url)
    html = result.body
    matches = PLAYER_LINK_PATTERN.findall(html)
    if not matches:
        return None
    return matches[0]


def fetch_player_gamelog_html(player_slug: str, season_end_year: int) -> str:
    base = settings.basketball_reference_base_url.rstrip("/")
    slug = player_slug.replace(".html", "")
    url = f"{base}{slug}/gamelog/{season_end_year}"
    client = _get_client()
    result = client.get(url)
    return result.body
