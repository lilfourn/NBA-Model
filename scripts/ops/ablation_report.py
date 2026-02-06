"""Ablation report: compare each expert and ensemble component on recent resolved data.

Outputs rolling metrics for each expert alone, meta-learner, online ensemble,
hybrid final, and p_final (post-shrink).  Flags when simpler components
outperform the full hybrid pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

EXPERT_COLS = ["p_forecast_cal", "p_nn", "p_tabdl", "p_lr", "p_xgb", "p_lgbm"]
ROLLING_WINDOW = 50


def _logloss(y: int, p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def _rolling_metrics(
    probs: np.ndarray, labels: np.ndarray, window: int = ROLLING_WINDOW
) -> dict:
    """Compute rolling accuracy and logloss over the last `window` rows."""
    if len(probs) < window:
        return {}
    recent_p = probs[-window:]
    recent_y = labels[-window:]
    picks = (recent_p >= 0.5).astype(int)
    accuracy = float((picks == recent_y).mean())
    avg_ll = float(
        np.mean([_logloss(int(y), float(p)) for y, p in zip(recent_y, recent_p)])
    )
    brier = float(np.mean((recent_p - recent_y) ** 2))
    return {
        "rolling_accuracy": round(accuracy, 4),
        "rolling_logloss": round(avg_ll, 4),
        "rolling_brier": round(brier, 4),
        "n": int(len(recent_p)),
    }


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

    labels_all = df["over_label"].to_numpy(dtype=float)
    components: dict[str, dict] = {}

    # Each expert alone
    for col in EXPERT_COLS:
        if col not in df.columns:
            continue
        valid_mask = df[col].notna() & df["over_label"].notna()
        if valid_mask.sum() < ROLLING_WINDOW:
            continue
        probs = df.loc[valid_mask, col].to_numpy(dtype=float)
        labels = df.loc[valid_mask, "over_label"].to_numpy(dtype=float)
        components[col] = _rolling_metrics(probs, labels)

    # p_final (post-shrink hybrid)
    if "p_final" in df.columns:
        valid_mask = df["p_final"].notna() & df["over_label"].notna()
        if valid_mask.sum() >= ROLLING_WINDOW:
            probs = df.loc[valid_mask, "p_final"].to_numpy(dtype=float)
            labels = df.loc[valid_mask, "over_label"].to_numpy(dtype=float)
            components["p_final"] = _rolling_metrics(probs, labels)

    # p_raw (pre-shrink hybrid)
    if "p_raw" in df.columns:
        valid_mask = df["p_raw"].notna() & df["over_label"].notna()
        if valid_mask.sum() >= ROLLING_WINDOW:
            probs = df.loc[valid_mask, "p_raw"].to_numpy(dtype=float)
            labels = df.loc[valid_mask, "over_label"].to_numpy(dtype=float)
            components["p_raw"] = _rolling_metrics(probs, labels)

    # Check if hybrid underperforms simpler components
    p_final_acc = components.get("p_final", {}).get("rolling_accuracy")
    warnings: list[str] = []
    if p_final_acc is not None:
        for name, m in components.items():
            if name in ("p_final", "p_raw"):
                continue
            comp_acc = m.get("rolling_accuracy")
            if comp_acc is not None and comp_acc > p_final_acc + 0.03:
                warnings.append(
                    f"{name} (acc={comp_acc:.1%}) outperforms p_final (acc={p_final_acc:.1%}) "
                    f"by {comp_acc - p_final_acc:.1%} -- possible integration bug"
                )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_back": days_back,
        "rolling_window": ROLLING_WINDOW,
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
        print(f"\nRolling {ROLLING_WINDOW}-bet ablation:")
        for name, m in sorted(
            components.items(),
            key=lambda x: x[1].get("rolling_accuracy", 0),
            reverse=True,
        ):
            print(
                f"  {name:<20} acc={m.get('rolling_accuracy', 'N/A')} ll={m.get('rolling_logloss', 'N/A')} brier={m.get('rolling_brier', 'N/A')}"
            )

    warnings = report.get("warnings", [])
    if warnings:
        print(f"\n{'!'*50}")
        for w in warnings:
            print(f"  WARNING: {w}")
        print(f"{'!'*50}")


if __name__ == "__main__":
    main()
