from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from app.db import schema
from app.ml.dataset import load_training_data
from app.ml.stat_mappings import stat_value_from_row
from app.ml.train import CATEGORICAL_COLS, MIN_TRAIN_ROWS, NUMERIC_COLS, _time_split
from app.modeling.conformal import ConformalCalibrator

XGBOOST_PARAMS = {
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "gamma": 0.1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "use_label_encoder": False,
    "verbosity": 0,
    "random_state": 42,
}


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int


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

    # One-hot encode categoricals for XGBoost (native categorical support is fragile
    # across joblib serialization, so we stay consistent with the LR pipeline).
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


def train_xgboost(engine, model_dir: Path) -> TrainResult:
    df = load_training_data(engine)
    if df.empty:
        raise RuntimeError("No training data available. Did you load NBA stats and build features?")

    X, y, df_used = _prepare_features(df)
    if df_used.empty:
        raise RuntimeError("Not enough training data after cleaning.")
    if y.nunique() < 2:
        raise RuntimeError("Not enough class variety to train yet.")
    if len(df_used) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Not enough training data available yet (rows={len(df_used)}).")

    X_train, X_test, y_train, y_test = _time_split(df_used, X, y)

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"XGB best iteration: {model.best_iteration} / {XGBOOST_PARAMS['n_estimators']}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    conformal = ConformalCalibrator.calibrate(y_proba, y_test.to_numpy(), alpha=0.10)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else None,
        "logloss": float(log_loss(y_test, y_proba)),
        "conformal_q_hat": conformal.q_hat,
        "conformal_n_cal": conformal.n_cal,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"xgb_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": list(X.columns),
            "categorical_cols": CATEGORICAL_COLS,
            "numeric_cols": NUMERIC_COLS,
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
                    "model_name": "xgboost",
                    "train_rows": int(len(df_used)),
                    "metrics": metrics,
                    "params": XGBOOST_PARAMS,
                    "artifact_path": str(model_path),
                }
            )
        )

    return TrainResult(model_path=str(model_path), metrics=metrics, rows=int(len(df_used)))
