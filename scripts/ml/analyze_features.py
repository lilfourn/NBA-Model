"""SHAP feature importance analysis for XGBoost model.

Computes SHAP values on holdout set, prints top features,
flags correlated pairs, and saves report to JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import load_training_data  # noqa: E402
from app.ml.stat_mappings import stat_value_from_row  # noqa: E402
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS, _time_split  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    if "is_combo" in df.columns:
        df = df[df["is_combo"].fillna(False) == False]  # noqa: E712
    if "player_name" in df.columns:
        df = df[~df["player_name"].fillna("").astype(str).str.contains("+", regex=False)]

    df["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in df.itertuples(index=False)
    ]
    df = df.dropna(subset=["line_score", "actual_value", "fetched_at"])
    if "minutes_to_start" in df.columns:
        df = df[df["minutes_to_start"].fillna(0) >= 0]
    if "is_live" in df.columns:
        df = df[df["is_live"].fillna(False) == False]  # noqa: E712
    if "in_game" in df.columns:
        df = df[df["in_game"].fillna(False) == False]  # noqa: E712

    df["over"] = (df["actual_value"] > df["line_score"]).astype(int)

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown").astype(str)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype(float)
    if "trending_count" in df.columns:
        df["trending_count"] = np.log1p(df["trending_count"].clip(lower=0.0))

    cat_dummies = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, dtype=float)
    X = pd.concat([cat_dummies, df[NUMERIC_COLS]], axis=1)
    y = df["over"]
    return X, y, df


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP feature importance analysis on XGBoost model.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--model-path", default=None, help="Path to XGB joblib. Auto-detects latest if not set.")
    ap.add_argument("--output", default="data/reports/feature_importance.json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--corr-threshold", type=float, default=0.8, help="Flag feature pairs with |r| above this.")
    ap.add_argument("--max-samples", type=int, default=2000, help="Max holdout samples for SHAP (speed).")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    # Load or auto-detect model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        models_dir = Path("models")
        candidates = sorted(models_dir.glob("xgb_*.joblib"))
        if not candidates:
            raise SystemExit("No XGBoost model found in models/")
        model_path = candidates[-1]
    print(f"Using model: {model_path}")

    payload = joblib.load(str(model_path))
    model = payload["model"]

    # Build holdout set
    df = load_training_data(engine)
    if df.empty:
        raise SystemExit("No training data.")
    X, y, df_used = _prepare_features(df)
    _, X_test, _, y_test = _time_split(df_used, X, y)

    if len(X_test) > args.max_samples:
        idx = np.random.default_rng(42).choice(len(X_test), args.max_samples, replace=False)
        X_test = X_test.iloc[idx]
        y_test = y_test.iloc[idx]

    print(f"Computing SHAP on {len(X_test)} holdout samples...")

    # Align columns to model's training features
    train_cols = payload.get("feature_cols", list(X.columns))
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[train_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(train_cols, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\nTop {args.top} features by mean |SHAP|:")
    print("-" * 50)
    for rank, (feat, imp) in enumerate(feature_importance[:args.top], 1):
        print(f"{rank:>3}. {feat:<35} {imp:.4f}")

    # Correlation analysis on numeric features only
    numeric_in_test = [c for c in NUMERIC_COLS if c in X_test.columns]
    corr_matrix = X_test[numeric_in_test].corr()
    high_corr_pairs = []
    for i in range(len(numeric_in_test)):
        for j in range(i + 1, len(numeric_in_test)):
            r = abs(corr_matrix.iloc[i, j])
            if r >= args.corr_threshold:
                f1, f2 = numeric_in_test[i], numeric_in_test[j]
                imp1 = dict(feature_importance).get(f1, 0.0)
                imp2 = dict(feature_importance).get(f2, 0.0)
                keep = f1 if imp1 >= imp2 else f2
                drop = f2 if keep == f1 else f1
                high_corr_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": round(float(r), 4),
                    "recommendation": f"keep {keep}, consider dropping {drop}",
                })

    if high_corr_pairs:
        print(f"\nHighly correlated pairs (|r| >= {args.corr_threshold}):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature_1']} <-> {pair['feature_2']}: r={pair['correlation']:.3f} -> {pair['recommendation']}")
    else:
        print(f"\nNo feature pairs with |r| >= {args.corr_threshold}")

    # Zero/near-zero importance features
    low_importance = [
        {"feature": f, "mean_abs_shap": round(imp, 6)}
        for f, imp in feature_importance
        if imp < 0.001
    ]
    if low_importance:
        print(f"\nLow-importance features (mean |SHAP| < 0.001): {[f['feature'] for f in low_importance]}")

    # Save report
    report = {
        "model_path": str(model_path),
        "holdout_samples": len(X_test),
        "feature_importance": [
            {"feature": f, "mean_abs_shap": round(imp, 6)}
            for f, imp in feature_importance
        ],
        "high_correlation_pairs": high_corr_pairs,
        "low_importance_features": low_importance,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {output}")


if __name__ == "__main__":
    main()
