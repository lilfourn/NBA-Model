from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.nba import fetch_nba_stats as mod


def test_dedupe_rows_by_game_and_player() -> None:
    rows = [
        {"GAME_ID": "g1", "PLAYER_ID": "p1", "PTS": 10},
        {"GAME_ID": "g1", "PLAYER_ID": "p1", "PTS": 12},
        {"GAME_ID": "g1", "PLAYER_ID": "p2", "PTS": 8},
    ]
    deduped = mod._dedupe_rows(rows)
    assert len(deduped) == 2
    assert any(r["PLAYER_ID"] == "p1" for r in deduped)
    assert any(r["PLAYER_ID"] == "p2" for r in deduped)


def test_fetch_chunked_by_day_partial_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    reset_calls: list[str] = []

    def fake_fetch(*, season: str, season_type: str, date_from: str, date_to: str):
        calls.append((date_from, date_to))
        if date_from == "2026-02-02":
            raise RuntimeError("timeout")
        return [{"GAME_ID": f"g-{date_from}", "PLAYER_ID": "p1"}]

    monkeypatch.setattr(mod, "reset_nba_stats_client", lambda: reset_calls.append("x"))
    monkeypatch.setattr(mod, "fetch_league_game_log", fake_fetch)

    rows = mod._fetch_chunked_by_day(
        season="2025-26",
        season_type="Regular Season",
        date_from="2026-02-01",
        date_to="2026-02-03",
    )
    assert len(rows) == 2
    assert calls == [
        ("2026-02-01", "2026-02-01"),
        ("2026-02-02", "2026-02-02"),
        ("2026-02-03", "2026-02-03"),
    ]
    assert len(reset_calls) == 3


def test_fetch_chunked_by_day_all_fail_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(*, season: str, season_type: str, date_from: str, date_to: str):
        raise RuntimeError("timeout")

    monkeypatch.setattr(mod, "reset_nba_stats_client", lambda: None)
    monkeypatch.setattr(mod, "fetch_league_game_log", fake_fetch)

    with pytest.raises(RuntimeError, match="failed for all days"):
        mod._fetch_chunked_by_day(
            season="2025-26",
            season_type="Regular Season",
            date_from="2026-02-01",
            date_to="2026-02-02",
        )


def test_main_reuses_existing_data_when_upstream_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(mod, "set_log_path", lambda _p: None)
    monkeypatch.setattr(mod, "log_run_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "fetch_league_game_log", lambda **_k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(mod, "_fetch_chunked_by_day", lambda **_k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(mod, "get_engine", lambda _url=None: SimpleNamespace(name="engine"))
    monkeypatch.setattr(mod, "_existing_stats_rows", lambda *_a, **_k: 123)

    def fake_load(rows, *, engine):
        calls["rows"] = rows
        calls["engine"] = engine
        return {"inserted": 0}

    monkeypatch.setattr(mod, "load_league_game_logs", fake_load)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "fetch_nba_stats.py",
            "--season",
            "2025-26",
            "--date-from",
            "2026-02-01",
            "--date-to",
            "2026-02-02",
        ],
    )
    mod.main()

    assert calls["rows"] == []
    assert getattr(calls["engine"], "name", "") == "engine"


def test_main_raises_when_upstream_fails_and_no_existing_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "set_log_path", lambda _p: None)
    monkeypatch.setattr(mod, "log_run_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "fetch_league_game_log", lambda **_k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(mod, "_fetch_chunked_by_day", lambda **_k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(mod, "get_engine", lambda _url=None: SimpleNamespace(name="engine"))
    monkeypatch.setattr(mod, "_existing_stats_rows", lambda *_a, **_k: 0)
    monkeypatch.setattr(mod, "load_league_game_logs", lambda *_a, **_k: {"inserted": 0})
    monkeypatch.setattr(
        mod.sys,
        "argv",
        [
            "fetch_nba_stats.py",
            "--season",
            "2025-26",
            "--date-from",
            "2026-02-01",
            "--date-to",
            "2026-02-02",
        ],
    )

    with pytest.raises(RuntimeError, match="no existing DB data"):
        mod.main()
