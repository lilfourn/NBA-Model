from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.artifacts import load_joblib_artifact
from app.ml.calibration import CalibratedExpert, PlattCalibrator, load_calibrator
from app.ml.dataset import _add_history_features
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS


@dataclass
class InferenceResult:
    frame: pd.DataFrame
    probs: np.ndarray
    model_path: str


def _load_model(path: str) -> tuple[Any, list[str], list[str], list[str], CalibratedExpert | PlattCalibrator | None]:
    payload = load_joblib_artifact(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unexpected model artifact format: {path}")
    feature_cols = list(payload.get("feature_cols") or [])
    cat_cols = list(payload.get("categorical_cols") or CATEGORICAL_COLS)
    num_cols = list(payload.get("numeric_cols") or NUMERIC_COLS)
    calibrator = None
    if "isotonic" in payload:
        try:
            calibrator = load_calibrator(payload["isotonic"])
        except Exception:  # noqa: BLE001
            pass
    return payload["model"], feature_cols, cat_cols, num_cols, calibrator


def _load_inference_frame(engine: Engine, snapshot_id: str) -> pd.DataFrame:
    query = text(
        """
        select
            pf.snapshot_id,
            pf.projection_id,
            pf.player_id,
            pf.game_id,
            pf.line_score,
            pf.line_score_prev,
            pf.line_score_delta,
            pf.line_movement,
            pf.stat_type,
            pf.projection_type,
            pf.odds_type,
            pf.trending_count,
            pf.is_promo,
            pf.is_live,
            pf.in_game,
            pf.today,
            pf.minutes_to_start,
            pf.fetched_at,
            pf.start_time,
            pl.name_key as prizepicks_name_key,
            pl.display_name as player_name,
            pl.combo as combo
        from projection_features pf
        join projections p
            on p.snapshot_id = pf.snapshot_id
            and p.projection_id = pf.projection_id
        join players pl on pl.id = pf.player_id
        where pf.snapshot_id = :snapshot_id
          and coalesce(p.odds_type, 0) = 0
          and lower(coalesce(p.event_type, p.attributes->>'event_type', '')) <> 'combo'
          and (pl.combo is null or pl.combo = false)
          and (pf.is_live is null or pf.is_live = false)
          and (pf.in_game is null or pf.in_game = false)
          and (pf.minutes_to_start is null or pf.minutes_to_start >= 0)
        """
    )
    return pd.read_sql(query, engine, params={"snapshot_id": snapshot_id})


def infer_over_probs(*, engine: Engine, model_path: str, snapshot_id: str) -> InferenceResult:
    frame = _load_inference_frame(engine, snapshot_id)
    if frame.empty:
        return InferenceResult(frame=frame, probs=np.zeros((0,), dtype=np.float32), model_path=model_path)

    model, train_feature_cols, cat_cols, num_cols, calibrator = _load_model(model_path)

    frame = frame.copy()
    frame = _add_history_features(frame, engine)

    for col in cat_cols:
        if col not in frame.columns:
            frame[col] = "unknown"
        frame[col] = frame[col].fillna("unknown").astype(str)

    for col in num_cols:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[num_cols] = frame[num_cols].fillna(0.0).astype(float)

    if "trending_count" in frame.columns:
        frame["trending_count"] = np.log1p(frame["trending_count"].clip(lower=0.0))

    cat_dummies = pd.get_dummies(frame[cat_cols], prefix=cat_cols, dtype=float)
    X = pd.concat([cat_dummies, frame[num_cols]], axis=1)

    # Align columns to training feature set (handles unseen categories).
    for col in train_feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[train_feature_cols]
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    probs = model.predict_proba(X)[:, 1].astype(np.float32)
    if calibrator is not None:
        probs = calibrator.transform(probs)
    return InferenceResult(frame=frame, probs=probs, model_path=model_path)
