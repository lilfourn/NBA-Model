"""Unified crawler client with retry, circuit breaker, rate limiter, and session management."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from curl_cffi import requests as curl_requests

# ---------------------------------------------------------------------------
# User-Agent pool — rotated per-request for fingerprint diversity.
# ---------------------------------------------------------------------------
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

IMPERSONATE_OPTIONS: list[str] = [
    "chrome",
    "chrome110",
    "chrome120",
]


def _pick_ua() -> str:
    return random.choice(USER_AGENTS)


def _pick_impersonate() -> str:
    return random.choice(IMPERSONATE_OPTIONS)


def _jittered_backoff(base: float, attempt: int, jitter_factor: float = 0.5) -> float:
    """Exponential backoff with random jitter to de-correlate retries."""
    delay = base * (2 ** attempt)
    jitter = delay * jitter_factor * random.random()
    return delay + jitter


# ---------------------------------------------------------------------------
# CrawlResult
# ---------------------------------------------------------------------------
@dataclass
class CrawlResult:
    status_code: int
    headers: dict[str, str]
    body: str
    json_data: Any | None
    source: str
    elapsed_ms: float
    attempt_count: int
    cached: bool = False


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
@dataclass
class CircuitBreaker:
    """Per-source circuit breaker to avoid hammering a down source."""
    failure_threshold: int = 5
    cooldown_seconds: float = 60.0

    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if (time.monotonic() - self._opened_at) >= self.cooldown_seconds:
                # half-open: allow one attempt
                self._opened_at = None
                self._failures = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
@dataclass
class RateLimiter:
    """Thread-safe token-bucket-style min-interval rate limiter."""
    min_interval_seconds: float = 1.0

    _next_allowed: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self.min_interval_seconds
                    return
                wait_for = self._next_allowed - now
            if wait_for > 0:
                time.sleep(wait_for)


# ---------------------------------------------------------------------------
# CrawlerClient
# ---------------------------------------------------------------------------
class CrawlerClient:
    """Unified HTTP client with retry, circuit breaker, rate limiter, and session reuse."""

    def __init__(
        self,
        *,
        source_name: str,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        connect_timeout: float = 5.0,
        read_timeout: float = 20.0,
        min_request_interval: float = 0.0,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        impersonate: str | None = None,
        proxy: str | None = None,
        default_headers: dict[str, str] | None = None,
        should_retry: Callable[[int], bool] | None = None,
        rotate_ua: bool = True,
        rotate_impersonate: bool = False,
    ) -> None:
        self.source_name = source_name
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.impersonate = impersonate or "chrome"
        self.proxy = proxy
        self.default_headers = dict(default_headers or {})
        self.should_retry = should_retry
        self.rotate_ua = rotate_ua
        self.rotate_impersonate = rotate_impersonate

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )
        self.rate_limiter = RateLimiter(min_interval_seconds=min_request_interval)

        # Session reuse — one per client instance.
        self._session: curl_requests.Session | None = None
        self._session_lock = threading.Lock()

    def _get_session(self) -> curl_requests.Session:
        with self._session_lock:
            if self._session is None:
                imp = _pick_impersonate() if self.rotate_impersonate else self.impersonate
                self._session = curl_requests.Session(impersonate=imp)
            return self._session

    def _reset_session(self) -> None:
        with self._session_lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:  # noqa: BLE001
                    pass
                self._session = None

    def _build_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = self.default_headers.copy()
        if self.rotate_ua:
            headers["User-Agent"] = _pick_ua()
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CrawlResult:
        """
        Make a GET request with retry, circuit breaker, and rate limiting.

        Raises the last exception if all retries are exhausted or circuit is open.
        """
        if self.circuit_breaker.is_open:
            raise CircuitOpenError(
                f"Circuit breaker open for {self.source_name}; "
                f"cooldown {self.circuit_breaker.cooldown_seconds}s"
            )

        effective_timeout = timeout or self.read_timeout
        merged_headers = self._build_headers(headers)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()

            start_time = time.monotonic()
            try:
                session = self._get_session()
                response = session.get(
                    url,
                    params=params or {},
                    headers=merged_headers,
                    timeout=effective_timeout,
                    proxy=self.proxy or None,
                )

                elapsed_ms = (time.monotonic() - start_time) * 1000

                if self.should_retry and self.should_retry(response.status_code):
                    raise RetryableStatusError(
                        f"Retryable status {response.status_code} from {self.source_name}"
                    )

                response.raise_for_status()
                self.circuit_breaker.record_success()

                json_data = None
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    try:
                        json_data = response.json()
                    except Exception:  # noqa: BLE001
                        pass

                return CrawlResult(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.text,
                    json_data=json_data,
                    source=self.source_name,
                    elapsed_ms=elapsed_ms,
                    attempt_count=attempt + 1,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.circuit_breaker.record_failure()

                if attempt >= self.max_retries:
                    break

                sleep_for = _jittered_backoff(self.backoff_seconds, attempt)
                time.sleep(sleep_for)

                # Rotate UA on retry for variety.
                if self.rotate_ua:
                    merged_headers["User-Agent"] = _pick_ua()

        # Reset session after exhausting retries (start fresh next time).
        self._reset_session()

        if last_error:
            raise last_error
        raise RuntimeError(f"{self.source_name} request failed without exception")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open for a source."""


class RetryableStatusError(RuntimeError):
    """Raised on a retryable HTTP status code to trigger retry logic."""
