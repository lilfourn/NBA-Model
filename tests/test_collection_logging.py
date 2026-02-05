"""Tests for app.clients.logging — structured collection logging and health report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.clients.logging import (
    generate_health_report,
    log_cache_hit,
    log_cache_miss,
    log_circuit_open,
    log_request_end,
    log_request_error,
    log_request_start,
    log_run_summary,
    log_validation,
    set_log_path,
)


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "logs" / "collection.jsonl"
    set_log_path(path)
    yield path


class TestLogEvents:
    def test_request_start(self, log_file):
        log_request_start("nba_stats", "https://api.example.com")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "request_start"
        assert entry["source"] == "nba_stats"

    def test_request_end(self, log_file):
        log_request_end("prizepicks", "https://api.example.com", status_code=200, elapsed_ms=123.4, attempt=1)
        entry = json.loads(log_file.read_text().strip())
        assert entry["event"] == "request_end"
        assert entry["status_code"] == 200
        assert entry["elapsed_ms"] == 123.4

    def test_request_error(self, log_file):
        log_request_error("nba_stats", "https://api.example.com", error="Timeout", attempt=2)
        entry = json.loads(log_file.read_text().strip())
        assert entry["event"] == "request_error"
        assert entry["error"] == "Timeout"

    def test_cache_events(self, log_file):
        log_cache_hit("nba_stats", "https://api.example.com")
        log_cache_miss("prizepicks", "https://api.example.com")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "cache_hit"
        assert json.loads(lines[1])["event"] == "cache_miss"

    def test_circuit_open(self, log_file):
        log_circuit_open("basketball_reference", 60.0)
        entry = json.loads(log_file.read_text().strip())
        assert entry["event"] == "circuit_open"
        assert entry["cooldown_seconds"] == 60.0

    def test_validation(self, log_file):
        log_validation("prizepicks", valid=False, errors=["empty data"])
        entry = json.loads(log_file.read_text().strip())
        assert entry["event"] == "validation"
        assert entry["valid"] is False
        assert "empty data" in entry["errors"]

    def test_run_summary(self, log_file):
        log_run_summary("nba_stats", duration_seconds=12.5, counts={"rows": 100})
        entry = json.loads(log_file.read_text().strip())
        assert entry["event"] == "run_summary"
        assert entry["counts"]["rows"] == 100

    def test_multiple_events_append(self, log_file):
        log_request_start("a", "url1")
        log_request_start("b", "url2")
        log_request_start("c", "url3")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3


class TestHealthReport:
    def test_empty_log(self, tmp_path):
        report = generate_health_report(tmp_path / "nonexistent.jsonl")
        assert "error" in report

    def test_basic_report(self, log_file):
        log_request_end("nba_stats", "url", status_code=200, elapsed_ms=100, attempt=1)
        log_request_end("nba_stats", "url", status_code=200, elapsed_ms=200, attempt=1)
        log_request_error("nba_stats", "url", error="timeout", attempt=1)
        log_cache_hit("nba_stats", "url")
        log_cache_miss("nba_stats", "url")
        log_cache_miss("nba_stats", "url2")

        report = generate_health_report(log_file, hours=1)
        stats = report["sources"]["nba_stats"]
        assert stats["requests"] == 2
        assert stats["errors"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["avg_elapsed_ms"] == 150.0
        assert stats["cache_hit_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_multi_source(self, log_file):
        log_request_end("nba_stats", "url", status_code=200, elapsed_ms=50, attempt=1)
        log_request_end("prizepicks", "url", status_code=200, elapsed_ms=80, attempt=1)
        log_circuit_open("basketball_reference", 60.0)

        report = generate_health_report(log_file, hours=1)
        assert "nba_stats" in report["sources"]
        assert "prizepicks" in report["sources"]
        assert report["sources"]["basketball_reference"]["circuit_opens"] == 1

    def test_filters_old_events(self, log_file):
        # Write an event with a timestamp > 24h ago by manually writing.
        old_entry = json.dumps({"event": "request_end", "source": "old", "ts": "2020-01-01T00:00:00+00:00", "status_code": 200, "elapsed_ms": 10, "attempt": 1})
        with open(log_file, "a") as f:
            f.write(old_entry + "\n")
        log_request_end("new", "url", status_code=200, elapsed_ms=10, attempt=1)

        report = generate_health_report(log_file, hours=24)
        assert "old" not in report["sources"]
        assert "new" in report["sources"]
