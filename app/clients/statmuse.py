from __future__ import annotations

from urllib.parse import quote

from app.clients.base import CrawlerClient
from app.clients.shared import get_shared_cache
from app.core.config import settings

_client: CrawlerClient | None = None


def _get_client() -> CrawlerClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = CrawlerClient(
            source_name="statmuse",
            max_retries=settings.statmuse_max_retries,
            backoff_seconds=settings.statmuse_backoff_seconds,
            read_timeout=float(settings.statmuse_timeout_seconds),
            impersonate=settings.nba_stats_impersonate,
            proxy=settings.statmuse_proxy or None,
            default_headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": settings.statmuse_base_url + "/",
            },
            should_retry=lambda status: status in {403, 429, 500, 502, 503, 504},
            min_request_interval=1.5,
            cache=get_shared_cache(),
        )
    return _client


def build_ask_url(query: str) -> str:
    base = settings.statmuse_base_url.rstrip("/")
    slug = quote(query.lower().strip())
    return f"{base}/nba/ask/{slug}"


def fetch_ask_html(query: str) -> str:
    url = build_ask_url(query)
    client = _get_client()
    result = client.get(url)
    return result.body


def build_player_gamelog_query(player_name: str, season_end_year: int) -> str:
    return f"{player_name} game log {season_end_year}"
