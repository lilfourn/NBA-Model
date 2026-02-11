from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import scripts.ops.monitor_model_health as monitor

from scripts.ops.monitor_model_health import _check_ensemble_weights
from scripts.ops.monitor_model_health import _check_rolling_metrics
from scripts.ops.monitor_model_health import ROLLING_WINDOW


def test_check_ensemble_weights_handles_context_nested_format(tmp_path: Path) -> None:
    path = tmp_path / "ensemble_weights.json"
    payload = {
        "weights": {
            '["PTS","pregame","neff_ge15"]': {"p_nn": 0.92, "p_lr": 0.08},
            '["AST","pregame","neff_ge15"]': {"p_nn": 0.90, "p_lr": 0.10},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    alerts = _check_ensemble_weights(str(path))
    assert any(
        a.get("type") == "weight_collapse" and a.get("expert") == "p_nn" for a in alerts
    )


def test_check_ensemble_weights_handles_flat_format(tmp_path: Path) -> None:
    path = tmp_path / "ensemble_weights.json"
    payload = {"weights": {"p_nn": 0.4, "p_lr": 0.3, "p_xgb": 0.3}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    alerts = _check_ensemble_weights(str(path))
    assert alerts == []


def test_check_rolling_metrics_prefers_is_correct_for_p_final() -> None:
    rows = []
    for idx in range(ROLLING_WINDOW):
        # Deliberately make p_final threshold disagree with labels.
        p_final = 0.9 if idx % 2 == 0 else 0.1
        label = 0 if idx % 2 == 0 else 1
        rows.append(
            {
                "p_final": p_final,
                "p_forecast_cal": 0.5,
                "p_nn": 0.5,
                "p_tabdl": 0.5,
                "p_lr": 0.5,
                "p_xgb": 0.5,
                "p_lgbm": 0.5,
                "over_label": label,
                "is_correct": 1.0,  # Ground-truth stored correctness
            }
        )
    df = pd.DataFrame(rows)

    result = _check_rolling_metrics(df)
    p_final_metrics = result["metrics"]["p_final"]
    assert p_final_metrics["rolling_accuracy"] == 1.0

    # Verify new diagnostic fields exist
    assert "base_rate" in p_final_metrics
    assert "inversion_test" in p_final_metrics
    assert "confusion_matrix" in p_final_metrics
    inv = p_final_metrics["inversion_test"]
    assert "accuracy" in inv
    assert "accuracy_inverted" in inv
    assert "logloss" in inv
    assert "logloss_inverted" in inv
    cm = p_final_metrics["confusion_matrix"]
    assert set(cm.keys()) == {"tp", "fp", "tn", "fn"}

    # Base rate should be returned at the top level of result
    assert "base_rate" in result
    assert result["base_rate"] is not None


def test_inversion_test_detects_inverted_predictions() -> None:
    """When p_final is consistently wrong, inversion test should flag it."""
    rows = []
    for idx in range(ROLLING_WINDOW):
        # p_final always predicts OVER (>0.5) but label is always UNDER (0)
        rows.append(
            {
                "p_final": 0.85,
                "p_forecast_cal": 0.85,
                "p_nn": 0.85,
                "p_tabdl": 0.85,
                "p_lr": 0.85,
                "p_xgb": 0.85,
                "p_lgbm": 0.85,
                "over_label": 0,
            }
        )
    df = pd.DataFrame(rows)

    result = _check_rolling_metrics(df)
    inv = result["metrics"]["p_final"]["inversion_test"]
    assert inv["accuracy"] == 0.0
    assert inv["accuracy_inverted"] == 1.0
    assert inv["inversion_improves_accuracy"] is True
    assert inv["inversion_improves_logloss"] is True


def test_build_health_report_has_live_metadata(monkeypatch) -> None:
    now = pd.Timestamp("2026-02-06T12:00:00Z")
    rows = []
    for idx in range(ROLLING_WINDOW):
        rows.append(
            {
                "p_final": 0.9,
                "p_forecast_cal": 0.8,
                "p_nn": 0.8,
                "p_tabdl": 0.8,
                "p_lr": 0.8,
                "p_xgb": 0.8,
                "p_lgbm": 0.8,
                "over_label": 1,
                "is_correct": 1.0,
                "event_time": now + pd.Timedelta(minutes=idx),
            }
        )
    df = pd.DataFrame(rows)

    monkeypatch.setattr(
        monitor,
        "_load_resolved_predictions",
        lambda _engine, days_back=90: df,
    )
    monkeypatch.setattr(
        monitor,
        "_load_ensemble_mean_weights",
        lambda _path: {"p_forecast_cal": 0.5, "p_nn": 0.5},
    )
    monkeypatch.setattr(monitor, "_check_ensemble_weights", lambda _path: [])

    report = monitor.build_health_report(
        engine=object(),
        days_back=90,
        ensemble_weights_path="models/ensemble_weights.json",
    )

    assert report["data_source"] == "live_db"
    assert report["generated_at"]
    assert report["latest_event_time"]
    assert report["prediction_log_rows"] == ROLLING_WINDOW
    assert "p_final" in report["expert_metrics"]
    assert "base_rate" in report
    assert report["base_rate"] is not None


def test_build_health_report_includes_collect_guardrail_alerts(monkeypatch) -> None:
    rows = []
    for _ in range(ROLLING_WINDOW):
        rows.append(
            {
                "p_final": 0.7,
                "p_forecast_cal": 0.7,
                "p_nn": 0.7,
                "p_tabdl": 0.7,
                "p_lr": 0.7,
                "p_xgb": 0.7,
                "p_lgbm": 0.7,
                "over_label": 1,
                "is_correct": 1.0,
            }
        )
    resolved_df = pd.DataFrame(rows)

    monkeypatch.setattr(
        monitor,
        "_load_resolved_predictions",
        lambda _engine, days_back=90: resolved_df,
    )
    monkeypatch.setattr(monitor, "_load_ensemble_mean_weights", lambda _path: {})
    monkeypatch.setattr(monitor, "_check_ensemble_weights", lambda _path: [])
    monkeypatch.setattr(
        monitor,
        "_compute_latest_collect_metrics",
        lambda _engine: {
            "snapshot_id": "snap-1",
            "total_scored": 100,
            "publishable_count": 10,
            "publishable_by_stat": {"Free Throws Made": 9, "Rebounds": 1},
            "top_stat_share": 0.9,
            "expert_coverage": {
                "p_forecast_cal": 1.0,
                "p_nn": 0.4,
                "p_tabdl": 0.95,
                "p_lr": 0.99,
                "p_xgb": 0.98,
                "p_lgbm": 0.97,
            },
        },
    )
    monkeypatch.setattr(
        monitor,
        "_load_recent_collect_summaries",
        lambda _engine, limit=5: [
            {
                "snapshot_id": "snap-1",
                "scored_at": "2026-02-11T12:00:00Z",
                "total_scored": 100,
                "publishable_count": 10,
                "publishable_ratio": 0.10,
            },
            {
                "snapshot_id": "snap-0",
                "scored_at": "2026-02-11T10:00:00Z",
                "total_scored": 120,
                "publishable_count": 12,
                "publishable_ratio": 0.10,
            },
        ],
    )
    monkeypatch.setattr(
        monitor,
        "_load_calibrator_flat_stat_types",
        lambda *_, **__: ["Assists"],
    )

    report = monitor.build_health_report(engine=object())

    assert report["publishable_by_stat"]["Free Throws Made"] == 9
    assert report["top_stat_share"] == 0.9
    assert report["expert_coverage"]["p_nn"] == 0.4
    assert report["calibrator_flat_stat_types"] == ["Assists"]

    alert_types = {a["type"] for a in report["alerts"]}
    assert "expert_coverage_low" in alert_types
    assert "low_publishable_ratio" in alert_types
    assert "calibrator_flat_stats" in alert_types
    suppressed_types = {a["type"] for a in report["suppressed_alerts"]}
    assert "stat_concentration" in suppressed_types
