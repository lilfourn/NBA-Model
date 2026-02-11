from __future__ import annotations

from typing import Any


from app.clients.base import CrawlerClient
from app.core.config import settings

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
}

_client: CrawlerClient | None = None


def _get_client() -> CrawlerClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = CrawlerClient(
            source_name="prizepicks",
            max_retries=2,
            backoff_seconds=1.5,
            read_timeout=float(settings.prizepicks_timeout_seconds),
            impersonate=settings.prizepicks_impersonate,
            default_headers=DEFAULT_HEADERS,
            min_request_interval=0.0,
        )
    return _client


def build_projections_url(base_url: str | None = None) -> str:
    api_base = (base_url or settings.prizepicks_api_url).rstrip("/")
    return f"{api_base}/projections"


def _build_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = DEFAULT_HEADERS.copy()
    headers["User-Agent"] = settings.prizepicks_user_agent
    if extra_headers:
        headers.update(extra_headers)
    return headers


def fetch_projections(
    *,
    league_id: int | None = None,
    per_page: int | None = None,
    timeout_seconds: int | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    effective_league_id = settings.prizepicks_league_id if league_id is None else league_id
    effective_per_page = settings.prizepicks_per_page if per_page is None else per_page
    effective_timeout = (
        settings.prizepicks_timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    params = {
        "league_id": effective_league_id,
        "per_page": effective_per_page,
    }
    url = build_projections_url()
    client = _get_client()
    result = client.get(
        url,
        params=params,
        headers=_build_headers(extra_headers),
        timeout=float(effective_timeout),
    )
    if result.json_data is not None:
        return result.json_data
    import json
    return json.loads(result.body)
