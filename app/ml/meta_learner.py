"""Stacked meta-learner: trains on OOF predictions from base experts.

Learns optimal blending weights from LR/XGB/LGBM OOF probabilities.
At inference time, takes live base-expert probabilities and outputs a
calibrated ensemble probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.db import schema

BASE_EXPERT_COLS = ["oof_lr", "oof_xgb", "oof_lgbm"]
INFER_EXPERT_COLS = ["p_lr", "p_xgb", "p_lgbm"]


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int
    coefficients: dict[str, float]


def train_meta_learner(
    *,
    oof_path: str = "data/oof_predictions.csv",
    model_dir: Path = Path("models"),
    engine=None,
) -> TrainResult:
    df = pd.read_csv(oof_path)
    if df.empty:
        raise RuntimeError(f"OOF predictions file is empty: {oof_path}")

    for col in BASE_EXPERT_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Missing column {col} in {oof_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=BASE_EXPERT_COLS + ["over"])
    if len(df) < 100:
        raise RuntimeError(f"Not enough OOF rows ({len(df)}).")

    X = df[BASE_EXPERT_COLS].values
    y = df["over"].astype(int).values

    # Time-ordered split: last 20% as holdout
    n = len(df)
    cutoff = int(n * 0.8)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else None,
        "logloss": float(log_loss(y_test, y_proba)),
    }

    # Extract learned blending weights
    lr_model = model.named_steps["lr"]
    coef = lr_model.coef_[0]
    coefficients = {col: round(float(c), 4) for col, c in zip(BASE_EXPERT_COLS, coef)}
    coefficients["intercept"] = round(float(lr_model.intercept_[0]), 4)

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"meta_learner_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "base_expert_cols": BASE_EXPERT_COLS,
            "infer_expert_cols": INFER_EXPERT_COLS,
            "coefficients": coefficients,
        },
        model_path,
    )

    if engine is not None:
        run_id = uuid4()
        with engine.begin() as conn:
            conn.execute(
                schema.model_runs.insert().values(
                    {
                        "id": run_id,
                        "created_at": datetime.now(timezone.utc),
                        "model_name": "meta_learner",
                        "train_rows": int(n),
                        "metrics": metrics,
                        "params": {"base_experts": BASE_EXPERT_COLS, "coefficients": coefficients},
                        "artifact_path": str(model_path),
                    }
                )
            )

    return TrainResult(
        model_path=str(model_path),
        metrics=metrics,
        rows=n,
        coefficients=coefficients,
    )


def infer_meta_learner(
    *,
    model_path: str,
    expert_probs: dict[str, float | None],
) -> float | None:
    """Run meta-learner on a single row's expert probabilities.

    Args:
        model_path: Path to saved meta-learner joblib.
        expert_probs: Dict with keys matching INFER_EXPERT_COLS.

    Returns:
        Meta-learner P(over) or None if any input is missing.
    """
    payload = joblib.load(model_path)
    model = payload["model"]
    infer_cols = payload.get("infer_expert_cols", INFER_EXPERT_COLS)

    values = []
    for col in infer_cols:
        v = expert_probs.get(col)
        if v is None:
            return None
        values.append(float(v))

    X = np.array([values])
    return float(model.predict_proba(X)[0, 1])


def batch_infer_meta_learner(
    *,
    model_path: str,
    expert_df: pd.DataFrame,
) -> np.ndarray:
    """Batch inference: expert_df columns must include INFER_EXPERT_COLS."""
    payload = joblib.load(model_path)
    model = payload["model"]
    infer_cols = payload.get("infer_expert_cols", INFER_EXPERT_COLS)

    for col in infer_cols:
        if col not in expert_df.columns:
            return np.full(len(expert_df), np.nan)

    X = expert_df[infer_cols].fillna(0.5).values
    return model.predict_proba(X)[:, 1].astype(np.float32)
