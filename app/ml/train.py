from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.db import schema
from app.ml.dataset import load_training_data
from app.ml.stat_mappings import stat_value_from_row
from app.modeling.conformal import ConformalCalibrator

MIN_TRAIN_ROWS = 50


CATEGORICAL_COLS = ["stat_type", "projection_type", "line_movement"]
NUMERIC_COLS = [
    "line_score",
    "line_score_prev",
    "line_score_delta",
    "minutes_to_start",
    "odds_type",
    "trending_count",
    "is_promo",
    "is_live",
    "in_game",
    "today",
    "hist_n",
    "hist_mean",
    "hist_std",
    "league_mean",
    "mu_stab",
    "p_hist_over",
    "z_line",
    "rest_days",
    "is_back_to_back",
    "stat_mean_3",
    "stat_mean_5",
    "stat_mean_10",
    "minutes_mean_3",
    "minutes_mean_5",
    "minutes_mean_10",
    "is_home",
    "opp_def_stat_avg",
    "opp_def_points_avg",
    "opp_def_rebounds_avg",
    "opp_def_assists_avg",
    "trend_slope",
    "stat_cv",
    "recent_vs_season",
    "minutes_trend",
    "stat_std_5",
    "line_move_pct",
    "line_move_late",
    "opp_def_rank",
    "stat_rate_per_min",
    "line_vs_mean_ratio",
    "hot_streak_count",
    "cold_streak_count",
    "season_game_number",
]


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    # Keep training aligned with inference: exclude PrizePicks combo players and "+" names.
    if "is_combo" in df.columns:
        df = df[df["is_combo"].fillna(False) == False]  # noqa: E712
    if "player_name" in df.columns:
        df = df[~df["player_name"].fillna("").astype(str).str.contains("+", regex=False)]

    df["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in df.itertuples(index=False)
    ]
    df = df.dropna(subset=["line_score", "actual_value", "fetched_at"])

    # Avoid training on in-game lines (different regime + leakage risk).
    if "minutes_to_start" in df.columns:
        df = df[df["minutes_to_start"].fillna(0) >= 0]
    if "is_live" in df.columns:
        df = df[df["is_live"].fillna(False) == False]  # noqa: E712
    if "in_game" in df.columns:
        df = df[df["in_game"].fillna(False) == False]  # noqa: E712

    df["over"] = (df["actual_value"] > df["line_score"]).astype(int)

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown").astype(str)
    # Coerce numerics to float to avoid object dtypes (Decimals) breaking sklearn.
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype(float)
    # Stabilize very large-count features.
    if "trending_count" in df.columns:
        df["trending_count"] = np.log1p(df["trending_count"].clip(lower=0.0))

    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = df["over"]
    return X, y, df


def _time_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, holdout_ratio: float = 0.2):
    df_sorted = df.sort_values("fetched_at")
    cutoff = int(len(df_sorted) * (1 - holdout_ratio))
    train_idx = df_sorted.index[:cutoff]
    test_idx = df_sorted.index[cutoff:]
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def train_baseline(engine, model_dir: Path) -> TrainResult:
    df = load_training_data(engine)
    if df.empty:
        raise RuntimeError("No training data available. Did you load NBA stats and build features?")

    X, y, df_used = _prepare_features(df)
    if df_used.empty:
        raise RuntimeError("Not enough training data after cleaning. Did you load NBA stats?")
    if df_used.empty:
        raise RuntimeError("Not enough training data available yet.")
    if y.nunique() < 2:
        raise RuntimeError("Not enough class variety to train yet.")
    if len(df_used) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Not enough training data available yet (rows={len(df_used)}).")
    X_train, X_test, y_train, y_test = _time_split(df_used, X, y)

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", SimpleImputer(strategy="constant", fill_value=0.0), NUMERIC_COLS),
        ],
    )
    model = LogisticRegression(max_iter=5000, solver="liblinear", C=0.1)
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    with warnings.catch_warnings():
        # Numpy/sklearn can emit noisy RuntimeWarnings during BLAS matmul on some builds
        # (even when outputs are finite). Suppress to keep training output actionable.
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Conformal calibration on holdout
    conformal = ConformalCalibrator.calibrate(y_proba, y_test.to_numpy(), alpha=0.10)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else None,
        "conformal_q_hat": conformal.q_hat,
        "conformal_n_cal": conformal.n_cal,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"baseline_logreg_{timestamp}.joblib"
    joblib.dump(
        {
            "model": pipeline,
            "feature_cols": {"categorical": CATEGORICAL_COLS, "numeric": NUMERIC_COLS},
            "conformal": {"alpha": conformal.alpha, "q_hat": conformal.q_hat, "n_cal": conformal.n_cal},
        },
        model_path,
    )

    run_id = uuid4()
    with engine.begin() as conn:
        conn.execute(
            schema.model_runs.insert().values(
                {
                    "id": run_id,
                    "created_at": datetime.now(timezone.utc),
                    "model_name": "baseline_logreg",
                    "train_rows": int(len(df_used)),
                    "metrics": metrics,
                    "params": {"model": "logistic_regression", "max_iter": 5000, "preprocess": "onehot+standardize"},
                    "artifact_path": str(model_path),
                }
            )
        )

    return TrainResult(model_path=str(model_path), metrics=metrics, rows=int(len(df_used)))
