from __future__ import annotations

import pandas as pd

from app.ml.drift_detection import DriftResult
from scripts.ops.check_drift import _apply_drift_alert_policy
from scripts.ops.check_drift import _split_row_fallback_windows


def _resolved_frame(rows: int) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "prob_over": [0.55] * rows,
            "actual_value": [12.0] * rows,
            "line_score": [10.0] * rows,
            "stat_type": ["Points"] * rows,
            "n_eff": [8.0] * rows,
            "event_time": times,
        }
    )


def test_row_fallback_balances_windows_when_recent_hint_is_too_large() -> None:
    frame = _resolved_frame(1000)
    recent, baseline, meta = _split_row_fallback_windows(
        frame,
        recent_hint_rows=5000,
        min_baseline_rows=100,
    )

    assert meta["reason"] == "ok"
    assert len(recent) == 500
    assert len(baseline) == 500
    assert recent["event_time"].min() > baseline["event_time"].max()


def test_row_fallback_returns_empty_when_total_rows_are_insufficient() -> None:
    frame = _resolved_frame(120)
    recent, baseline, meta = _split_row_fallback_windows(
        frame,
        recent_hint_rows=200,
        min_baseline_rows=100,
    )

    assert recent.empty
    assert baseline.empty
    assert meta["reason"] == "not_enough_total_rows"


def _drift(check_type: str, is_drifted: bool) -> DriftResult:
    return DriftResult(
        check_type=check_type,
        is_drifted=is_drifted,
        metric_value=0.5,
        threshold=0.2,
        details={},
    )


def test_alert_policy_downgrades_distribution_only_in_row_fallback() -> None:
    results = [_drift("distribution", True)]
    actionable, suppressed, policy = _apply_drift_alert_policy(
        results,
        window_mode="row_fallback",
        fallback_distribution_actionable=False,
    )
    assert policy == "fallback_distribution_advisory"
    assert actionable == []
    assert len(suppressed) == 1
    assert suppressed[0]["check_type"] == "distribution"


def test_alert_policy_keeps_non_distribution_drift_actionable() -> None:
    results = [_drift("distribution", True), _drift("performance", True)]
    actionable, suppressed, policy = _apply_drift_alert_policy(
        results,
        window_mode="row_fallback",
        fallback_distribution_actionable=False,
    )
    assert policy == "fallback_distribution_advisory"
    assert len(actionable) == 2
    assert suppressed == []


def test_alert_policy_can_force_strict_fallback_distribution() -> None:
    results = [_drift("distribution", True)]
    actionable, suppressed, policy = _apply_drift_alert_policy(
        results,
        window_mode="row_fallback",
        fallback_distribution_actionable=True,
    )
    assert policy == "fallback_strict"
    assert len(actionable) == 1
    assert suppressed == []
