"""Tests for app.clients.cache — ResponseCache."""
from __future__ import annotations

import time

import pytest

from app.clients.cache import ResponseCache


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(cache_dir=str(tmp_path / "cache"), default_ttl_seconds=60.0)


class TestCacheHitMiss:
    def test_miss_returns_none(self, cache):
        assert cache.get("nba_stats", "https://example.com/api") is None

    def test_put_then_get(self, cache):
        cache.put(
            "nba_stats", "https://example.com/api", {"season": "2024"},
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"data": []}',
            json_data={"data": []},
        )
        entry = cache.get("nba_stats", "https://example.com/api", {"season": "2024"})
        assert entry is not None
        assert entry.status_code == 200
        assert entry.json_data == {"data": []}
        assert entry.body == '{"data": []}'

    def test_different_params_miss(self, cache):
        cache.put(
            "nba_stats", "https://example.com/api", {"season": "2024"},
            status_code=200, headers={}, body="ok",
        )
        assert cache.get("nba_stats", "https://example.com/api", {"season": "2025"}) is None

    def test_different_source_miss(self, cache):
        cache.put(
            "nba_stats", "https://example.com/api", None,
            status_code=200, headers={}, body="ok",
        )
        assert cache.get("basketball_reference", "https://example.com/api") is None


class TestCacheExpiry:
    def test_expired_returns_none(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), default_ttl_seconds=0.05)
        cache.put("src", "https://example.com", None, status_code=200, headers={}, body="ok")
        time.sleep(0.06)
        assert cache.get("src", "https://example.com") is None

    def test_stale_while_revalidate(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), default_ttl_seconds=0.05)
        cache.put("src", "https://example.com", None, status_code=200, headers={}, body="stale")
        time.sleep(0.06)
        assert cache.get("src", "https://example.com") is None
        stale = cache.get_stale("src", "https://example.com")
        assert stale is not None
        assert stale.body == "stale"


class TestCachePerSourceTTL:
    def test_custom_ttl(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), default_ttl_seconds=0.05)
        cache.set_ttl("long_ttl", 60.0)
        cache.put("long_ttl", "https://example.com", None, status_code=200, headers={}, body="ok")
        time.sleep(0.06)
        # Should still be valid because custom TTL is 60s.
        assert cache.get("long_ttl", "https://example.com") is not None


class TestConditionalHeaders:
    def test_returns_etag(self, cache):
        cache.put(
            "src", "https://example.com", None,
            status_code=200,
            headers={"ETag": '"abc123"'},
            body="ok",
        )
        headers = cache.get_conditional_headers("src", "https://example.com")
        assert headers.get("If-None-Match") == '"abc123"'

    def test_returns_last_modified(self, cache):
        cache.put(
            "src", "https://example.com", None,
            status_code=200,
            headers={"Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
            body="ok",
        )
        headers = cache.get_conditional_headers("src", "https://example.com")
        assert headers.get("If-Modified-Since") == "Wed, 21 Oct 2015 07:28:00 GMT"

    def test_no_entry_returns_empty(self, cache):
        assert cache.get_conditional_headers("src", "https://nope.com") == {}


class TestCacheClear:
    def test_clear_source(self, cache):
        cache.put("a", "https://a.com", None, status_code=200, headers={}, body="a")
        cache.put("b", "https://b.com", None, status_code=200, headers={}, body="b")
        removed = cache.clear("a")
        assert removed == 1
        assert cache.get("a", "https://a.com") is None
        assert cache.get("b", "https://b.com") is not None

    def test_clear_all(self, cache):
        cache.put("a", "https://a.com", None, status_code=200, headers={}, body="a")
        cache.put("b", "https://b.com", None, status_code=200, headers={}, body="b")
        removed = cache.clear()
        assert removed == 2

    def test_purge_older_than(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), default_ttl_seconds=60.0)
        cache.put("src", "https://old.com", None, status_code=200, headers={}, body="old")
        time.sleep(0.05)
        cache.put("src", "https://new.com", None, status_code=200, headers={}, body="new")
        removed = cache.purge_older_than(0.03)
        assert removed == 1
        assert cache.get("src", "https://new.com") is not None
