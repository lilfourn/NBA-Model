"""Run drift detection checks and output a JSON report.

Called from cron_train.sh after training. Exits with code 1 if any
drift is detected (can be used to trigger conditional retraining).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.drift_detection import (  # noqa: E402
    DriftResult,
    check_calibration_drift,
    check_distribution_drift,
    check_performance_drift,
)
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EVENT_TIME_SQL = "coalesce(pp.decision_time, pp.resolved_at, pp.created_at)"
CANONICAL_VIEW = "vw_resolved_picks_canonical"
METRICS_VERSION = "2.0.0"  # Bump when drift detection logic changes


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for col in ["prob_over", "actual_value", "line_score", "n_eff"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "event_time" in out.columns:
        out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce", utc=True)
    return out


def _load_recent_predictions(engine, days_back: int = 7) -> pd.DataFrame:
    """Load recent resolved predictions (canonical view with fallback)."""
    canonical_sql = text(
        f"""
        select
            pp.p_final as prob_over, pp.actual_value,
            pp.line_at_decision as line_score, pp.stat_type, pp.n_eff,
            coalesce(pp.decision_time, pp.resolved_at, pp.created_at) as event_time
        from {CANONICAL_VIEW} pp
        where coalesce(pp.decision_time, pp.resolved_at, pp.created_at)
              >= now() - make_interval(days => :days_back)
        order by event_time desc
    """
    )
    fallback_sql = text(
        f"""
        select
            pp.prob_over, pp.actual_value, pp.line_score,
            pp.stat_type, pp.n_eff,
            {EVENT_TIME_SQL} as event_time
        from projection_predictions pp
        where pp.actual_value is not null
          and {EVENT_TIME_SQL} >= now() - make_interval(days => :days_back)
        order by event_time desc
    """
    )
    try:
        try:
            frame = pd.read_sql(
                canonical_sql, engine, params={"days_back": int(max(1, days_back))}
            )
        except Exception:  # noqa: BLE001
            frame = pd.read_sql(
                fallback_sql, engine, params={"days_back": int(max(1, days_back))}
            )
        return _coerce_types(frame)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _load_baseline_predictions(
    engine, days_back: int = 30, offset_days: int = 7
) -> pd.DataFrame:
    """Load baseline predictions (older window, canonical view with fallback)."""
    params = {
        "total_days": int(max(1, days_back + offset_days)),
        "offset_days": int(max(1, offset_days)),
    }
    canonical_sql = text(
        f"""
        select
            pp.p_final as prob_over, pp.actual_value,
            pp.line_at_decision as line_score, pp.stat_type, pp.n_eff,
            coalesce(pp.decision_time, pp.resolved_at, pp.created_at) as event_time
        from {CANONICAL_VIEW} pp
        where coalesce(pp.decision_time, pp.resolved_at, pp.created_at)
              >= now() - make_interval(days => :total_days)
          and coalesce(pp.decision_time, pp.resolved_at, pp.created_at)
              < now() - make_interval(days => :offset_days)
        order by event_time desc
    """
    )
    fallback_sql = text(
        f"""
        select
            pp.prob_over, pp.actual_value, pp.line_score,
            pp.stat_type, pp.n_eff,
            {EVENT_TIME_SQL} as event_time
        from projection_predictions pp
        where pp.actual_value is not null
          and {EVENT_TIME_SQL} >= now() - make_interval(days => :total_days)
          and {EVENT_TIME_SQL} < now() - make_interval(days => :offset_days)
        order by event_time desc
    """
    )
    try:
        try:
            frame = pd.read_sql(canonical_sql, engine, params=params)
        except Exception:  # noqa: BLE001
            frame = pd.read_sql(fallback_sql, engine, params=params)
        return _coerce_types(frame)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _load_resolved_predictions_for_fallback(
    engine, *, max_rows: int = 50000
) -> pd.DataFrame:
    """Load resolved predictions for row-based fallback splitting.

    Uses event-time ordering so the fallback remains chronological.
    """
    query = text(
        f"""
        select
            pp.prob_over,
            pp.actual_value,
            pp.line_score,
            pp.stat_type,
            pp.n_eff,
            {EVENT_TIME_SQL} as event_time
        from projection_predictions pp
        where pp.actual_value is not null
          and pp.prob_over is not null
          and pp.line_score is not null
          and {EVENT_TIME_SQL} is not null
        order by event_time desc
        limit :max_rows
        """
    )
    try:
        frame = pd.read_sql(query, engine, params={"max_rows": int(max(200, max_rows))})
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if frame.empty:
        return frame
    frame = _coerce_types(frame)
    return frame.sort_values("event_time").reset_index(drop=True)


def _split_row_fallback_windows(
    frame: pd.DataFrame,
    *,
    recent_hint_rows: int,
    min_baseline_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    """Build recent/baseline windows from row counts when time windows fail."""
    min_rows = int(max(30, min_baseline_rows))
    if frame.empty:
        return frame, frame, {"reason": "empty_frame", "total_rows": 0}

    ordered = frame.sort_values("event_time").reset_index(drop=True)
    total_rows = len(ordered)
    if total_rows < (min_rows * 2):
        return (
            ordered.iloc[0:0].copy(),
            ordered.iloc[0:0].copy(),
            {"reason": "not_enough_total_rows", "total_rows": int(total_rows)},
        )

    # Keep the fallback balanced so baseline has enough signal.
    recent_rows = min(max(min_rows, int(recent_hint_rows)), total_rows // 2)
    baseline_rows = min(recent_rows, total_rows - recent_rows)
    if baseline_rows < min_rows:
        return (
            ordered.iloc[0:0].copy(),
            ordered.iloc[0:0].copy(),
            {"reason": "not_enough_baseline_rows", "total_rows": int(total_rows)},
        )

    baseline_start = total_rows - recent_rows - baseline_rows
    baseline_end = total_rows - recent_rows
    baseline = ordered.iloc[baseline_start:baseline_end].copy()
    recent = ordered.iloc[baseline_end:].copy()
    return (
        recent,
        baseline,
        {
            "reason": "ok",
            "total_rows": int(total_rows),
            "recent_rows": int(len(recent)),
            "baseline_rows": int(len(baseline)),
        },
    )


def _apply_drift_alert_policy(
    results: list[DriftResult],
    *,
    window_mode: str,
    fallback_distribution_actionable: bool,
) -> tuple[list[DriftResult], list[dict[str, str]], str]:
    """Return actionable drift checks after applying operational alert policy."""
    raw_drifted = [r for r in results if r.is_drifted]
    if not raw_drifted:
        return [], [], "standard"

    if window_mode != "row_fallback":
        return raw_drifted, [], "standard"
    if fallback_distribution_actionable:
        return raw_drifted, [], "fallback_strict"

    non_distribution = [r for r in raw_drifted if r.check_type != "distribution"]
    if non_distribution:
        return raw_drifted, [], "fallback_distribution_advisory"

    suppressed = [
        {
            "check_type": r.check_type,
            "reason": "distribution_only_drift_in_row_fallback_mode",
        }
        for r in raw_drifted
    ]
    return [], suppressed, "fallback_distribution_advisory"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift detection checks")
    parser.add_argument(
        "--recent-days", type=int, default=7, help="Days for recent window"
    )
    parser.add_argument(
        "--baseline-days", type=int, default=30, help="Days for baseline window"
    )
    parser.add_argument(
        "--min-baseline-rows",
        type=int,
        default=200,
        help="Minimum baseline rows required before using row-fallback windows.",
    )
    parser.add_argument(
        "--fallback-max-rows",
        type=int,
        default=50000,
        help="Maximum resolved rows loaded for row-fallback splitting.",
    )
    parser.add_argument(
        "--disable-row-fallback",
        action="store_true",
        help="Disable row-based fallback when baseline time-window rows are insufficient.",
    )
    parser.add_argument(
        "--fallback-distribution-actionable",
        action="store_true",
        help=(
            "In row-fallback mode, treat distribution-only drift as actionable "
            "(default: advisory only to reduce noisy retrain triggers)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reports/drift_report.json",
        help="Output path",
    )
    args = parser.parse_args()

    load_env()
    engine = get_engine()

    recent = _load_recent_predictions(engine, days_back=args.recent_days)
    baseline = _load_baseline_predictions(
        engine, days_back=args.baseline_days, offset_days=args.recent_days
    )
    window_mode = "time_window"
    notes: list[str] = []
    fallback_info: dict[str, int | str] | None = None

    baseline_too_small = baseline.empty or len(baseline) < int(
        max(1, args.min_baseline_rows)
    )
    if baseline_too_small and not bool(args.disable_row_fallback):
        fallback_pool = _load_resolved_predictions_for_fallback(
            engine,
            max_rows=int(max(200, args.fallback_max_rows)),
        )
        recent_fb, baseline_fb, fallback_info = _split_row_fallback_windows(
            fallback_pool,
            recent_hint_rows=max(len(recent), int(max(1, args.min_baseline_rows))),
            min_baseline_rows=int(max(1, args.min_baseline_rows)),
        )
        if not recent_fb.empty and not baseline_fb.empty:
            recent = recent_fb
            baseline = baseline_fb
            window_mode = "row_fallback"
            notes.append(
                "Time-window baseline was insufficient; switched to row-based fallback windows."
            )
        else:
            notes.append(
                "Time-window baseline was insufficient and row-fallback could not produce valid windows."
            )
    elif baseline_too_small:
        notes.append("Baseline rows are insufficient and row-fallback is disabled.")

    results = []

    if not recent.empty and not baseline.empty:
        # Performance drift
        recent["correct"] = (
            (
                (recent["prob_over"] >= 0.5)
                & (recent["actual_value"] > recent["line_score"])
            )
            | (
                (recent["prob_over"] < 0.5)
                & (recent["actual_value"] <= recent["line_score"])
            )
        ).astype(float)
        baseline["correct"] = (
            (
                (baseline["prob_over"] >= 0.5)
                & (baseline["actual_value"] > baseline["line_score"])
            )
            | (
                (baseline["prob_over"] < 0.5)
                & (baseline["actual_value"] <= baseline["line_score"])
            )
        ).astype(float)

        results.append(
            check_performance_drift(
                recent["correct"].values, baseline["correct"].values
            )
        )

        # Distribution drift on key features
        dist_features = ["prob_over", "line_score"]
        if "n_eff" in recent.columns and "n_eff" in baseline.columns:
            dist_features.append("n_eff")

        baseline_feats = {}
        current_feats = {}
        for feat in dist_features:
            if feat in recent.columns and feat in baseline.columns:
                b_vals = pd.to_numeric(baseline[feat], errors="coerce").dropna().values
                c_vals = pd.to_numeric(recent[feat], errors="coerce").dropna().values
                if len(b_vals) > 10 and len(c_vals) > 10:
                    baseline_feats[feat] = b_vals
                    current_feats[feat] = c_vals

        if baseline_feats:
            results.append(check_distribution_drift(baseline_feats, current_feats))

        # Calibration drift
        probs = pd.to_numeric(recent["prob_over"], errors="coerce").dropna().values
        labels = (recent["actual_value"] > recent["line_score"]).astype(float).values
        if len(probs) >= 30:
            results.append(check_calibration_drift(probs, labels))

    insufficient_data = recent.empty or baseline.empty
    invalid_baseline = baseline.empty and not recent.empty
    if insufficient_data and not notes:
        notes.append("Insufficient recent/baseline rows to run drift checks.")
    if invalid_baseline:
        notes.append(
            "INVALID BASELINE: baseline has 0 rows after all fallback attempts."
        )

    # Data freshness gate: warn if the most recent decision_time is stale
    data_freshness_status = "ok"
    latest_event_time = None
    if not recent.empty and "event_time" in recent.columns:
        event_times = pd.to_datetime(recent["event_time"], errors="coerce").dropna()
        if not event_times.empty:
            latest = event_times.max()
            latest_event_time = str(latest)
            if latest.tzinfo is None:
                latest = latest.tz_localize("UTC")
            staleness_hours = (
                pd.Timestamp.now(tz="UTC") - latest
            ).total_seconds() / 3600
            if staleness_hours > 48:
                data_freshness_status = "stale"
                notes.append(
                    f"DATA STALE: latest decision_time is "
                    f"{staleness_hours:.0f}h old (>48h threshold)."
                )

    actionable_drift, suppressed_checks, alert_policy = _apply_drift_alert_policy(
        results,
        window_mode=window_mode,
        fallback_distribution_actionable=bool(args.fallback_distribution_actionable),
    )
    any_drift_raw = any(r.is_drifted for r in results)
    any_drift_actionable = len(actionable_drift) > 0
    if suppressed_checks:
        notes.append(
            "Distribution-only drift in row-fallback mode was downgraded to advisory "
            "to reduce noisy retrain triggers."
        )

    # Determine overall status
    if invalid_baseline:
        status = "invalid_baseline"
    elif data_freshness_status == "stale":
        status = "stale_data"
    elif any_drift_actionable:
        status = "drift_detected"
    elif insufficient_data:
        status = "insufficient_data"
    else:
        status = "ok"

    report = {
        "metrics_version": METRICS_VERSION,
        "status": status,
        "recent_rows": len(recent),
        "baseline_rows": len(baseline),
        "latest_event_time": latest_event_time,
        "data_freshness": data_freshness_status,
        "time_column": EVENT_TIME_SQL,
        "window_mode": window_mode,
        "alert_policy": alert_policy,
        "insufficient_data": bool(insufficient_data),
        "invalid_baseline": bool(invalid_baseline),
        "notes": notes,
        "checks": [r.to_dict() for r in results],
        "suppressed_checks": suppressed_checks,
        "any_drift_raw": any_drift_raw,
        "any_drift": any_drift_actionable,
    }
    if fallback_info is not None:
        report["fallback_info"] = fallback_info

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Drift report -> {out_path}")

    if invalid_baseline:
        print(
            "  WARNING: invalid_baseline -- baseline has 0 rows after all fallback attempts."
        )
    if data_freshness_status == "stale":
        print(f"  WARNING: stale data -- latest event is {latest_event_time}")
    if any_drift_actionable:
        for d in actionable_drift:
            print(
                f"  DRIFT DETECTED: {d.check_type} "
                f"(value={d.metric_value:.4f}, threshold={d.threshold})"
            )
        sys.exit(1)
    else:
        if insufficient_data:
            print("  Drift checks skipped: insufficient baseline/recent data.")
        elif any_drift_raw and suppressed_checks:
            print(
                "  Drift signals detected but downgraded to advisory by alert policy."
            )
        else:
            print("  No drift detected.")


if __name__ == "__main__":
    main()
