"""Run drift detection checks and output a JSON report.

Called from cron_train.sh after training. Exits with code 1 if any
drift is detected (can be used to trigger conditional retraining).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.drift_detection import (  # noqa: E402
    check_calibration_drift,
    check_distribution_drift,
    check_performance_drift,
    run_all_drift_checks,
)
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _load_recent_predictions(engine, days_back: int = 7) -> pd.DataFrame:
    """Load recent resolved predictions from projection_predictions."""
    query = text(
        """
        select
            pp.prob_over,
            pp.actual_value,
            pp.line_score,
            pp.stat_type,
            pp.n_eff,
            pp.created_at
        from projection_predictions pp
        where pp.actual_value is not null
          and pp.created_at >= now() - make_interval(days => :days_back)
        order by pp.created_at desc
        """
    )
    try:
        return pd.read_sql(query, engine, params={"days_back": days_back})
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _load_baseline_predictions(engine, days_back: int = 30, offset_days: int = 7) -> pd.DataFrame:
    """Load baseline predictions (older window for comparison)."""
    query = text(
        """
        select
            pp.prob_over,
            pp.actual_value,
            pp.line_score,
            pp.stat_type,
            pp.n_eff,
            pp.created_at
        from projection_predictions pp
        where pp.actual_value is not null
          and pp.created_at >= now() - make_interval(days => :total_days)
          and pp.created_at < now() - make_interval(days => :offset_days)
        order by pp.created_at desc
        """
    )
    try:
        return pd.read_sql(
            query, engine,
            params={"total_days": days_back + offset_days, "offset_days": offset_days},
        )
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift detection checks")
    parser.add_argument("--recent-days", type=int, default=7, help="Days for recent window")
    parser.add_argument("--baseline-days", type=int, default=30, help="Days for baseline window")
    parser.add_argument("--output", type=str, default="data/reports/drift_report.json", help="Output path")
    args = parser.parse_args()

    load_env()
    engine = get_engine()

    recent = _load_recent_predictions(engine, days_back=args.recent_days)
    baseline = _load_baseline_predictions(engine, days_back=args.baseline_days, offset_days=args.recent_days)

    results = []

    if not recent.empty and not baseline.empty:
        # Performance drift
        recent["correct"] = (
            ((recent["prob_over"] >= 0.5) & (recent["actual_value"] > recent["line_score"]))
            | ((recent["prob_over"] < 0.5) & (recent["actual_value"] <= recent["line_score"]))
        ).astype(float)
        baseline["correct"] = (
            ((baseline["prob_over"] >= 0.5) & (baseline["actual_value"] > baseline["line_score"]))
            | ((baseline["prob_over"] < 0.5) & (baseline["actual_value"] <= baseline["line_score"]))
        ).astype(float)

        results.append(
            check_performance_drift(recent["correct"].values, baseline["correct"].values)
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

    report = {
        "recent_rows": len(recent),
        "baseline_rows": len(baseline),
        "checks": [r.to_dict() for r in results],
        "any_drift": any(r.is_drifted for r in results),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Drift report -> {out_path}")

    any_drifted = any(r.is_drifted for r in results)
    if any_drifted:
        drifted = [r for r in results if r.is_drifted]
        for d in drifted:
            print(f"  DRIFT DETECTED: {d.check_type} (value={d.metric_value:.4f}, threshold={d.threshold})")
        sys.exit(1)
    else:
        print("  No drift detected.")


if __name__ == "__main__":
    main()
