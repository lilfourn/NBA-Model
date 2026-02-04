from __future__ import annotations

import time
from typing import Any, Callable

from curl_cffi import requests as curl_requests


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
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = curl_requests.get(
                url,
                params=params or {},
                headers=headers or {},
                timeout=timeout,
                impersonate=impersonate,
                proxy=proxy,
            )
            if should_retry and should_retry(response.status_code):
                raise RuntimeError(f"Retryable status {response.status_code}")
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "text": response.text,
                "json": (response.json() if "application/json" in response.headers.get("Content-Type", "") else None),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_for = backoff_seconds * (2**attempt)
            time.sleep(sleep_for)

    if last_error:
        raise last_error
    raise RuntimeError("Request failed without exception")
