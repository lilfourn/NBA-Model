"""Stacked meta-learner: trains on OOF predictions from base experts.

Learns optimal blending weights from all 5 expert OOF probabilities plus
context features. At inference time, takes live base-expert probabilities
and outputs a calibrated ensemble probability.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from app.db import schema  # noqa: E402
from app.ml.artifacts import load_joblib_artifact  # noqa: E402

BASE_EXPERT_COLS = ["oof_forecast", "oof_lr", "oof_xgb", "oof_lgbm", "oof_nn"]
INFER_EXPERT_COLS = ["p_forecast_cal", "p_lr", "p_xgb", "p_lgbm", "p_nn"]
CONTEXT_COLS = ["n_eff_log"]


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int
    coefficients: dict[str, float]


def _build_feature_matrix(df: pd.DataFrame, expert_cols: list[str]) -> np.ndarray:
    """Build feature matrix from expert probs + context columns."""
    parts = [df[expert_cols].fillna(0.5).values]
    # Log-scaled n_eff context
    if "n_eff" in df.columns:
        n_eff_log = np.log1p(pd.to_numeric(df["n_eff"], errors="coerce").fillna(0.0).values)
    elif "n_eff_log" in df.columns:
        n_eff_log = pd.to_numeric(df["n_eff_log"], errors="coerce").fillna(0.0).values
    else:
        n_eff_log = np.zeros(len(df))
    parts.append(n_eff_log.reshape(-1, 1))
    return np.hstack(parts)


def train_meta_learner(
    *,
    oof_path: str = "data/oof_predictions.csv",
    model_dir: Path = Path("models"),
    engine=None,
) -> TrainResult:
    df = pd.read_csv(oof_path)
    if df.empty:
        raise RuntimeError(f"OOF predictions file is empty: {oof_path}")

    # Support both old (3-expert) and new (4-expert) OOF files
    available_experts = [c for c in BASE_EXPERT_COLS if c in df.columns]
    if len(available_experts) < 3:
        raise RuntimeError(f"Need >= 3 expert columns, found {available_experts} in {oof_path}")

    for col in available_experts:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=available_experts + ["over"])
    if len(df) < 100:
        raise RuntimeError(f"Not enough OOF rows ({len(df)}).")

    X = _build_feature_matrix(df, available_experts)
    y = df["over"].astype(int).values

    # Map OOF col names to inference col names
    oof_to_infer = {
        "oof_forecast": "p_forecast_cal",
        "oof_nn": "p_nn",
        "oof_lr": "p_lr",
        "oof_xgb": "p_xgb",
        "oof_lgbm": "p_lgbm",
    }
    infer_cols = [oof_to_infer.get(c, c) for c in available_experts]
    feature_names = infer_cols + list(CONTEXT_COLS)

    # Time-ordered split: last 20% as holdout
    n = len(df)
    cutoff = int(n * 0.8)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    base_lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("cal", CalibratedClassifierCV(base_lr, cv=3, method="isotonic")),
    ])
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else None,
        "logloss": float(log_loss(y_test, y_proba)),
    }

    # Extract coefficients from base estimators for diagnostics
    coefficients: dict[str, float] = {}
    try:
        cal_model = model.named_steps["cal"]
        for i, est in enumerate(cal_model.calibrated_classifiers_):
            base = est.estimator
            if hasattr(base, "coef_"):
                for j, name in enumerate(feature_names):
                    if j < len(base.coef_[0]):
                        key = f"cv{i}_{name}"
                        coefficients[key] = round(float(base.coef_[0][j]), 4)
    except Exception:  # noqa: BLE001
        pass

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"meta_learner_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "base_expert_cols": available_experts,
            "infer_expert_cols": infer_cols,
            "context_cols": list(CONTEXT_COLS),
            "feature_names": feature_names,
            "coefficients": coefficients,
        },
        model_path,
    )

    if engine is not None:
        try:
            from app.ml.artifact_store import upload_file
            upload_file(engine, model_name="meta_learner", file_path=model_path)
            print(f"Uploaded meta_learner artifact to DB ({model_path})")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: DB upload failed for meta_learner: {exc}")

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
                        "params": {"base_experts": available_experts, "coefficients": coefficients},
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
    n_eff: float | None = None,
) -> float | None:
    """Run meta-learner on a single row's expert probabilities.

    Args:
        model_path: Path to saved meta-learner joblib.
        expert_probs: Dict with keys matching INFER_EXPERT_COLS.
        n_eff: Effective sample size for context.

    Returns:
        Meta-learner P(over) or None if too few experts available.
    """
    payload = load_joblib_artifact(model_path)
    model = payload["model"]
    infer_cols = payload.get("infer_expert_cols", INFER_EXPERT_COLS)
    has_context = bool(payload.get("context_cols"))

    # Require at least 3 expert values to produce a prediction
    values = []
    available_count = 0
    for col in infer_cols:
        v = expert_probs.get(col)
        if v is not None:
            values.append(float(v))
            available_count += 1
        else:
            values.append(0.5)  # fill missing with neutral
    if available_count < 3:
        return None

    if has_context:
        n_eff_log = float(np.log1p(n_eff)) if n_eff is not None and n_eff > 0 else 0.0
        values.append(n_eff_log)

    X = np.array([values])
    return float(model.predict_proba(X)[0, 1])


def batch_infer_meta_learner(
    *,
    model_path: str,
    expert_df: pd.DataFrame,
    n_eff_series: pd.Series | None = None,
) -> np.ndarray:
    """Batch inference: expert_df columns must include INFER_EXPERT_COLS."""
    payload = load_joblib_artifact(model_path)
    model = payload["model"]
    infer_cols = payload.get("infer_expert_cols", INFER_EXPERT_COLS)
    has_context = bool(payload.get("context_cols"))

    parts = [expert_df[infer_cols].fillna(0.5).values]
    if has_context:
        if n_eff_series is not None:
            n_eff_log = np.log1p(n_eff_series.fillna(0.0).values)
        else:
            n_eff_log = np.zeros(len(expert_df))
        parts.append(n_eff_log.reshape(-1, 1))

    X = np.hstack(parts)
    return model.predict_proba(X)[:, 1].astype(np.float32)
