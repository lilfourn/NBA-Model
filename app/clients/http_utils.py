from __future__ import annotations

import time
from typing import Any, Callable

from curl_cffi import requests as curl_requests

from app.clients.base import CrawlerClient

# Per-source client singletons keyed by (source_name).
_CLIENTS: dict[str, CrawlerClient] = {}


def _get_or_create_client(
    *,
    source_name: str,
    max_retries: int,
    backoff_seconds: float,
    timeout: int | float | None,
    impersonate: str | None,
    proxy: str | None,
    should_retry: Callable[[int], bool] | None,
    headers: dict[str, str] | None,
) -> CrawlerClient:
    if source_name not in _CLIENTS:
        _CLIENTS[source_name] = CrawlerClient(
            source_name=source_name,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            read_timeout=float(timeout or 20),
            impersonate=impersonate,
            proxy=proxy,
            default_headers=headers or {},
            should_retry=should_retry,
        )
    return _CLIENTS[source_name]


def get_with_retries(
    *,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int | float | None = None,
    impersonate: str | None = None,
    proxy: str | None = None,
    max_retries: int = 2,
    backoff_seconds: float = 1.5,
    should_retry: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper that delegates to CrawlerClient."""
    client = _get_or_create_client(
        source_name="http_utils",
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        timeout=timeout,
        impersonate=impersonate,
        proxy=proxy,
        should_retry=should_retry,
        headers=headers,
    )
    result = client.get(url, params=params, headers=headers, timeout=float(timeout or 20))
    return {
        "status_code": result.status_code,
        "headers": result.headers,
        "text": result.body,
        "json": result.json_data,
    }
