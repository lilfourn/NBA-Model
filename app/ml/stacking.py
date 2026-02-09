from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

EXPERT_COLS = ["p_lr", "p_xgb", "p_lgbm", "p_nn", "p_forecast_cal", "p_tabdl"]


def _logit(p: float) -> float:
    eps = 1e-7
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


@dataclass
class StackingResult:
    model: Any
    metrics: dict[str, Any]
    weights: dict[str, float]


def train_stacking_meta(oof_df: pd.DataFrame) -> StackingResult:
    """Train meta-learner on OOF predictions from base models."""
    df = oof_df.copy()
    for col in EXPERT_COLS:
        if col not in df.columns:
            df[col] = 0.5

    X = np.column_stack([df[col].apply(_logit).values for col in EXPERT_COLS])
    y = df["over"].values

    model = LogisticRegression(C=0.1, max_iter=500)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
    }
    weights = {col: float(w) for col, w in zip(EXPERT_COLS, model.coef_[0])}

    return StackingResult(model=model, metrics=metrics, weights=weights)


def predict_stacking(model: Any, expert_probs: dict[str, float | None]) -> float:
    """Predict using stacking meta-model."""
    features = []
    for col in EXPERT_COLS:
        p = expert_probs.get(col)
        features.append(_logit(p) if p is not None else 0.0)

    return float(model.predict_proba([features])[0, 1])
