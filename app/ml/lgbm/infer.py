from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.dataset import load_training_data
from app.ml.stat_mappings import stat_value_from_row
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS


@dataclass
class InferenceResult:
    frame: Any
    probs: np.ndarray


def _prepare_inference_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
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
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    return X[feature_cols]


def infer_over_probs(
    *,
    engine,
    model_path: str,
    snapshot_id: str,
) -> InferenceResult:
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    df = load_training_data(engine, snapshot_id=snapshot_id)
    if df.empty:
        return InferenceResult(frame=df, probs=np.zeros((0,), dtype=np.float32))

    df = df.copy()
    if "is_combo" in df.columns:
        df = df[df["is_combo"].fillna(False) == False]  # noqa: E712
    if "player_name" in df.columns:
        df = df[~df["player_name"].fillna("").astype(str).str.contains("+", regex=False)]
    if "minutes_to_start" in df.columns:
        df = df[df["minutes_to_start"].fillna(0) >= 0]
    if "is_live" in df.columns:
        df = df[df["is_live"].fillna(False) == False]  # noqa: E712
    if "in_game" in df.columns:
        df = df[df["in_game"].fillna(False) == False]  # noqa: E712

    X = _prepare_inference_features(df, feature_cols)
    probs = model.predict_proba(X)[:, 1].astype(np.float32)
    return InferenceResult(frame=df, probs=probs)
