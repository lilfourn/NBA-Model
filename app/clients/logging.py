"""Structured JSON logging for collection events.

Writes one JSON line per event to a configurable log file. Events include
request start/end, retries, circuit breaker state changes, cache hits/misses,
and validation results.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOG_PATH: Path | None = None
_LOCK = threading.Lock()


def set_log_path(path: str | Path) -> None:
    global _LOG_PATH  # noqa: PLW0603
    _LOG_PATH = Path(path)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _emit(event: dict[str, Any]) -> None:
    if _LOG_PATH is None:
        return
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(event, default=str)
    with _LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Request lifecycle events
# ---------------------------------------------------------------------------
def log_request_start(source: str, url: str, attempt: int = 1) -> None:
    _emit({"event": "request_start", "source": source, "url": url, "attempt": attempt})


def log_request_end(
    source: str,
    url: str,
    *,
    status_code: int,
    elapsed_ms: float,
    attempt: int,
    cached: bool = False,
) -> None:
    _emit({
        "event": "request_end",
        "source": source,
        "url": url,
        "status_code": status_code,
        "elapsed_ms": round(elapsed_ms, 1),
        "attempt": attempt,
        "cached": cached,
    })


def log_request_error(source: str, url: str, *, error: str, attempt: int) -> None:
    _emit({"event": "request_error", "source": source, "url": url, "error": error, "attempt": attempt})


# ---------------------------------------------------------------------------
# Circuit breaker events
# ---------------------------------------------------------------------------
def log_circuit_open(source: str, cooldown_seconds: float) -> None:
    _emit({"event": "circuit_open", "source": source, "cooldown_seconds": cooldown_seconds})


def log_circuit_half_open(source: str) -> None:
    _emit({"event": "circuit_half_open", "source": source})


# ---------------------------------------------------------------------------
# Cache events
# ---------------------------------------------------------------------------
def log_cache_hit(source: str, url: str) -> None:
    _emit({"event": "cache_hit", "source": source, "url": url})


def log_cache_miss(source: str, url: str) -> None:
    _emit({"event": "cache_miss", "source": source, "url": url})


def log_cache_stale(source: str, url: str) -> None:
    _emit({"event": "cache_stale", "source": source, "url": url})


# ---------------------------------------------------------------------------
# Validation events
# ---------------------------------------------------------------------------
def log_validation(source: str, *, valid: bool, errors: list[str] | None = None, warnings: list[str] | None = None) -> None:
    _emit({
        "event": "validation",
        "source": source,
        "valid": valid,
        "errors": errors or [],
        "warnings": warnings or [],
    })


# ---------------------------------------------------------------------------
# Collection run summary
# ---------------------------------------------------------------------------
def log_run_summary(source: str, *, duration_seconds: float, counts: dict[str, int], errors: list[str] | None = None) -> None:
    _emit({
        "event": "run_summary",
        "source": source,
        "duration_seconds": round(duration_seconds, 2),
        "counts": counts,
        "errors": errors or [],
    })


# ---------------------------------------------------------------------------
# Health report generation
# ---------------------------------------------------------------------------
def generate_health_report(log_path: str | Path, hours: int = 24) -> dict[str, Any]:
    """Parse the JSONL log and produce a summary report."""
    path = Path(log_path)
    if not path.exists():
        return {"error": "Log file not found", "path": str(path)}

    cutoff = time.time() - (hours * 3600)
    by_source: dict[str, dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_str = entry.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
            except (ValueError, TypeError):
                continue
            if ts < cutoff:
                continue

            source = entry.get("source", "unknown")
            if source not in by_source:
                by_source[source] = {
                    "requests": 0,
                    "errors": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "circuit_opens": 0,
                    "validation_failures": 0,
                    "total_elapsed_ms": 0.0,
                    "last_error": None,
                }

            stats = by_source[source]
            event = entry.get("event")

            if event == "request_end":
                stats["requests"] += 1
                stats["total_elapsed_ms"] += entry.get("elapsed_ms", 0)
            elif event == "request_error":
                stats["errors"] += 1
                stats["last_error"] = entry.get("error")
            elif event == "cache_hit":
                stats["cache_hits"] += 1
            elif event == "cache_miss":
                stats["cache_misses"] += 1
            elif event == "circuit_open":
                stats["circuit_opens"] += 1
            elif event == "validation":
                if not entry.get("valid"):
                    stats["validation_failures"] += 1

    # Compute derived metrics.
    for source, stats in by_source.items():
        total = stats["requests"]
        stats["avg_elapsed_ms"] = round(stats["total_elapsed_ms"] / total, 1) if total else 0
        cache_total = stats["cache_hits"] + stats["cache_misses"]
        stats["cache_hit_rate"] = round(stats["cache_hits"] / cache_total, 3) if cache_total else 0
        stats["error_rate"] = round(stats["errors"] / (total + stats["errors"]), 3) if (total + stats["errors"]) else 0

    return {
        "hours": hours,
        "sources": by_source,
    }
