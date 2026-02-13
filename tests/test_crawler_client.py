"""Tests for app.clients.base — CrawlerClient, CircuitBreaker, RateLimiter."""
from __future__ import annotations

import time

import pytest

from app.clients.base import (
    CircuitBreaker,
    CircuitOpenError,
    CrawlerClient,
    RateLimiter,
    USER_AGENTS,
    _jittered_backoff,
    _pick_ua,
)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------
class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert not cb.is_open

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.06)
        assert not cb.is_open  # half-open

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert not cb.is_open


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------
class TestRateLimiter:
    def test_no_delay_when_zero_interval(self):
        rl = RateLimiter(min_interval_seconds=0.0)
        start = time.monotonic()
        rl.wait()
        rl.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05

    def test_respects_min_interval(self):
        rl = RateLimiter(min_interval_seconds=0.1)
        start = time.monotonic()
        rl.wait()
        rl.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.09  # at least ~100ms between


# ---------------------------------------------------------------------------
# Jittered backoff
# ---------------------------------------------------------------------------
class TestJitteredBackoff:
    def test_increases_with_attempt(self):
        d0 = _jittered_backoff(1.0, 0, jitter_factor=0.0)
        d1 = _jittered_backoff(1.0, 1, jitter_factor=0.0)
        d2 = _jittered_backoff(1.0, 2, jitter_factor=0.0)
        assert d0 == pytest.approx(1.0)
        assert d1 == pytest.approx(2.0)
        assert d2 == pytest.approx(4.0)

    def test_jitter_adds_randomness(self):
        delays = {_jittered_backoff(1.0, 1, jitter_factor=0.5) for _ in range(20)}
        # With jitter, we should get at least a few different values.
        assert len(delays) > 1

    def test_stays_within_bounds(self):
        for _ in range(100):
            d = _jittered_backoff(1.0, 2, jitter_factor=0.5)
            # base=4.0, jitter up to 2.0 → max 6.0
            assert 4.0 <= d <= 6.0


# ---------------------------------------------------------------------------
# UA rotation
# ---------------------------------------------------------------------------
class TestUARotation:
    def test_pick_ua_returns_valid(self):
        ua = _pick_ua()
        assert ua in USER_AGENTS

    def test_pick_ua_produces_variety(self):
        uas = {_pick_ua() for _ in range(50)}
        assert len(uas) > 1


# ---------------------------------------------------------------------------
# CrawlerClient (unit-level, no real HTTP)
# ---------------------------------------------------------------------------
class TestCrawlerClientCircuitBreaker:
    def test_raises_circuit_open_error(self):
        client = CrawlerClient(
            source_name="test",
            max_retries=0,
            failure_threshold=1,
            cooldown_seconds=60.0,
        )
        # Trip the breaker.
        client.circuit_breaker.record_failure()
        assert client.circuit_breaker.is_open
        with pytest.raises(CircuitOpenError):
            client.get("http://localhost:99999/nope")


# ---------------------------------------------------------------------------
# PrizePicks client wiring
# ---------------------------------------------------------------------------
class TestPrizePicksClientRetry:
    def test_429_retry_callback(self):
        from app.clients.prizepicks import _get_client, _client
        import app.clients.prizepicks as pp_mod

        prev = pp_mod._client
        pp_mod._client = None
        try:
            client = _get_client()
            assert client.should_retry is not None
            assert client.should_retry(429) is True
            assert client.should_retry(200) is False
            assert client.should_retry(500) is False
            assert client.backoff_seconds == 5.0
        finally:
            pp_mod._client = prev
