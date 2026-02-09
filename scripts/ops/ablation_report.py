"""Ablation report: compare each expert and ensemble component on recent resolved data.

Outputs rolling metrics for each expert alone, meta-learner, online ensemble,
hybrid final, and p_final (post-shrink).  Flags when simpler components
outperform the full hybrid pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.ensemble_strategies import (
    ROLLING_WINDOWS,
    multi_window_metrics,
)  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]


def _load_data(engine, *, days_back: int = 90) -> pd.DataFrame:
    """Load resolved predictions with all expert columns."""
    from sqlalchemy import text as sa_text

    sql = sa_text(
        """
        select
            prob_over as p_final, p_raw,
            p_forecast_cal, p_nn,
            coalesce(p_tabdl::text, details->>'p_tabdl') as p_tabdl,
            p_lr, p_xgb, p_lgbm,
            over_label,
            coalesce(decision_time, resolved_at, created_at) as event_time
        from projection_predictions
        where over_label is not null
          and actual_value is not null
          and outcome in ('over', 'under')
          and coalesce(decision_time, resolved_at, created_at) >= now() - (:days_back * interval '1 day')
        order by coalesce(decision_time, resolved_at, created_at) asc
    """
    )
    df = pd.read_sql(sql, engine, params={"days_back": int(max(1, days_back))})
    for col in EXPERT_COLS + ["p_final", "p_raw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["over_label"] = pd.to_numeric(df["over_label"], errors="coerce")
    return df


def build_ablation_report(engine, *, days_back: int = 90) -> dict:
    """Build comparative ablation metrics for all model components."""
    df = _load_data(engine, days_back=days_back)
    if df.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "components": {},
        }

    components: dict[str, dict] = {}
    min_window = min(ROLLING_WINDOWS)

    for col in EXPERT_COLS:
        if col not in df.columns:
            continue
        valid_mask = df[col].notna() & df["over_label"].notna()
        if valid_mask.sum() < min_window:
            continue
        probs = df.loc[valid_mask, col].to_numpy(dtype=float)
        labels = df.loc[valid_mask, "over_label"].to_numpy(dtype=float)
        components[col] = multi_window_metrics(probs, labels)

    for col in ("p_final", "p_raw"):
        if col not in df.columns:
            continue
        valid_mask = df[col].notna() & df["over_label"].notna()
        if valid_mask.sum() < min_window:
            continue
        probs = df.loc[valid_mask, col].to_numpy(dtype=float)
        labels = df.loc[valid_mask, "over_label"].to_numpy(dtype=float)
        components[col] = multi_window_metrics(probs, labels)

    warnings: list[str] = []
    p_final_data = components.get("p_final", {})
    for window_key in [f"last_{w}" for w in ROLLING_WINDOWS]:
        p_final_acc = p_final_data.get(window_key, {}).get("rolling_accuracy")
        if p_final_acc is None:
            continue
        for name, m in components.items():
            if name in ("p_final", "p_raw"):
                continue
            comp_acc = m.get(window_key, {}).get("rolling_accuracy")
            if comp_acc is not None and comp_acc > p_final_acc + 0.03:
                warnings.append(
                    f"{name} ({window_key} acc={comp_acc:.1%}) outperforms p_final ({p_final_acc:.1%}) "
                    f"by {comp_acc - p_final_acc:.1%}"
                )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_back": days_back,
        "rolling_windows": ROLLING_WINDOWS,
        "total_rows": len(df),
        "components": components,
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate ablation report.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--days-back", type=int, default=90)
    ap.add_argument("--output", default="data/reports/ablation_report.json")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)
    report = build_ablation_report(engine, days_back=args.days_back)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Ablation report -> {output}")

    components = report.get("components", {})
    if components:
        for window_key in [f"last_{w}" for w in ROLLING_WINDOWS] + ["all_time"]:
            print(f"\n{window_key} ablation:")
            ranked = sorted(
                components.items(),
                key=lambda x: x[1].get(window_key, {}).get("rolling_accuracy", 0),
                reverse=True,
            )
            for name, m in ranked:
                wm = m.get(window_key, {})
                if not wm:
                    continue
                print(
                    f"  {name:<20} acc={wm.get('rolling_accuracy', 'N/A')} "
                    f"ll={wm.get('rolling_logloss', 'N/A')} brier={wm.get('rolling_brier', 'N/A')}"
                )

    warnings = report.get("warnings", [])
    if warnings:
        print(f"\n{'!'*50}")
        for w in warnings:
            print(f"  WARNING: {w}")
        print(f"{'!'*50}")


if __name__ == "__main__":
    main()
