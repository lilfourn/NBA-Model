"""File-based HTTP response cache with per-source TTL and conditional requests."""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    status_code: int
    headers: dict[str, str]
    body: str
    json_data: Any | None
    cached_at: float
    etag: str | None = None
    last_modified: str | None = None


class ResponseCache:
    """Gzipped JSON file cache keyed by source + url + params."""

    def __init__(self, cache_dir: str = "data/cache", default_ttl_seconds: float = 3600.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.default_ttl_seconds = default_ttl_seconds
        self._ttls: dict[str, float] = {}
        self._lock = threading.Lock()

    def set_ttl(self, source: str, ttl_seconds: float) -> None:
        self._ttls[source] = ttl_seconds

    def _ttl_for(self, source: str) -> float:
        return self._ttls.get(source, self.default_ttl_seconds)

    def _cache_key(self, source: str, url: str, params: dict[str, Any] | None = None) -> str:
        parts = f"{source}|{url}"
        if params:
            sorted_params = sorted(params.items())
            parts += "|" + "&".join(f"{k}={v}" for k, v in sorted_params)
        return hashlib.sha256(parts.encode()).hexdigest()

    def _cache_path(self, source: str, key: str) -> Path:
        source_dir = self.cache_dir / source
        return source_dir / f"{key}.json.gz"

    def get(self, source: str, url: str, params: dict[str, Any] | None = None) -> CacheEntry | None:
        key = self._cache_key(source, url, params)
        path = self._cache_path(source, key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.loads(f.read())
        except Exception:  # noqa: BLE001
            return None

        entry = CacheEntry(
            status_code=data.get("status_code", 200),
            headers=data.get("headers", {}),
            body=data.get("body", ""),
            json_data=data.get("json_data"),
            cached_at=data.get("cached_at", 0.0),
            etag=data.get("etag"),
            last_modified=data.get("last_modified"),
        )

        ttl = self._ttl_for(source)
        age = time.time() - entry.cached_at
        if age > ttl:
            return None  # Expired — caller should re-fetch.

        return entry

    def get_stale(self, source: str, url: str, params: dict[str, Any] | None = None) -> CacheEntry | None:
        """Get entry even if expired (for stale-while-revalidate on failure)."""
        key = self._cache_key(source, url, params)
        path = self._cache_path(source, key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.loads(f.read())
        except Exception:  # noqa: BLE001
            return None
        return CacheEntry(
            status_code=data.get("status_code", 200),
            headers=data.get("headers", {}),
            body=data.get("body", ""),
            json_data=data.get("json_data"),
            cached_at=data.get("cached_at", 0.0),
            etag=data.get("etag"),
            last_modified=data.get("last_modified"),
        )

    def get_conditional_headers(self, source: str, url: str, params: dict[str, Any] | None = None) -> dict[str, str]:
        """Return If-None-Match / If-Modified-Since headers if we have a cached entry."""
        key = self._cache_key(source, url, params)
        path = self._cache_path(source, key)
        if not path.exists():
            return {}
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.loads(f.read())
        except Exception:  # noqa: BLE001
            return {}
        headers: dict[str, str] = {}
        if data.get("etag"):
            headers["If-None-Match"] = data["etag"]
        if data.get("last_modified"):
            headers["If-Modified-Since"] = data["last_modified"]
        return headers

    def put(
        self,
        source: str,
        url: str,
        params: dict[str, Any] | None,
        *,
        status_code: int,
        headers: dict[str, str],
        body: str,
        json_data: Any | None = None,
    ) -> None:
        key = self._cache_key(source, url, params)
        path = self._cache_path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "status_code": status_code,
            "headers": headers,
            "body": body,
            "json_data": json_data,
            "cached_at": time.time(),
            "etag": headers.get("ETag") or headers.get("etag"),
            "last_modified": headers.get("Last-Modified") or headers.get("last-modified"),
        }

        with self._lock:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(json.dumps(entry))

    def clear(self, source: str | None = None) -> int:
        """Remove cache entries. If source given, only that source. Returns count removed."""
        count = 0
        if source:
            source_dir = self.cache_dir / source
            if source_dir.exists():
                for f in source_dir.glob("*.json.gz"):
                    f.unlink()
                    count += 1
        else:
            if self.cache_dir.exists():
                for f in self.cache_dir.rglob("*.json.gz"):
                    f.unlink()
                    count += 1
        return count

    def purge_older_than(self, max_age_seconds: float) -> int:
        """Remove entries older than max_age_seconds. Returns count removed."""
        count = 0
        cutoff = time.time() - max_age_seconds
        if not self.cache_dir.exists():
            return 0
        for path in self.cache_dir.rglob("*.json.gz"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    count += 1
            except OSError:
                pass
        return count
