from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.jobs import JobType
from app.services.scoring import ScoringResult


def _dummy_result() -> ScoringResult:
    return ScoringResult(
        snapshot_id="snapshot-1",
        scored_at="2026-02-09T00:00:00+00:00",
        total_scored=1,
        picks=[],
    )


def test_modal_db_source_uses_logged_predictions(monkeypatch) -> None:
    import app.api.picks as picks_api

    monkeypatch.setattr(picks_api.settings, "picks_source", "modal_db")
    monkeypatch.setattr(picks_api, "get_engine", lambda: object())

    calls = {"logged": 0, "ensemble": 0}

    def fake_logged(*_args, **_kwargs):
        calls["logged"] += 1
        return _dummy_result()

    def fake_ensemble(*_args, **_kwargs):
        calls["ensemble"] += 1
        return _dummy_result()

    monkeypatch.setattr(picks_api, "score_logged_predictions", fake_logged)
    monkeypatch.setattr(picks_api, "score_ensemble", fake_ensemble)
    monkeypatch.setattr(picks_api.job_manager, "is_running", lambda _jt: True)

    client = TestClient(app)
    resp = client.get("/api/picks?top=5")
    assert resp.status_code == 200
    assert calls["logged"] == 1
    assert calls["ensemble"] == 0


def test_modal_db_force_triggers_collect_job(monkeypatch) -> None:
    import app.api.picks as picks_api

    monkeypatch.setattr(picks_api.settings, "picks_source", "modal_db")
    monkeypatch.setattr(picks_api, "get_engine", lambda: object())
    monkeypatch.setattr(picks_api, "score_logged_predictions", lambda *_args, **_kwargs: _dummy_result())

    started: list[JobType] = []
    monkeypatch.setattr(picks_api.job_manager, "is_running", lambda _jt: False)
    monkeypatch.setattr(picks_api.job_manager, "start_job", lambda jt: started.append(jt))

    client = TestClient(app)
    resp = client.get("/api/picks?force=true")
    assert resp.status_code == 200
    assert started == [JobType.COLLECT]


def test_inline_source_uses_inline_scoring(monkeypatch) -> None:
    import app.api.picks as picks_api

    monkeypatch.setattr(picks_api.settings, "picks_source", "inline")
    monkeypatch.setattr(picks_api, "get_engine", lambda: object())

    calls = {"logged": 0, "ensemble": 0}

    def fake_logged(*_args, **_kwargs):
        calls["logged"] += 1
        return _dummy_result()

    def fake_ensemble(*_args, **_kwargs):
        calls["ensemble"] += 1
        return _dummy_result()

    monkeypatch.setattr(picks_api, "score_logged_predictions", fake_logged)
    monkeypatch.setattr(picks_api, "score_ensemble", fake_ensemble)

    client = TestClient(app)
    resp = client.get("/api/picks")
    assert resp.status_code == 200
    assert calls["logged"] == 0
    assert calls["ensemble"] == 1
