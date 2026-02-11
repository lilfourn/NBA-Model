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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.db import schema
from app.ml.calibration import best_calibrator
from app.ml.dataset import load_training_data
from app.ml.prepare_features import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    prepare_lr_features,
    time_series_cv_split,
)
from app.modeling.conformal import ConformalCalibrator

MIN_TRAIN_ROWS = 50


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int


def _time_split(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, holdout_ratio: float = 0.2
):
    df_sorted = df.sort_values("fetched_at")
    cutoff = int(len(df_sorted) * (1 - holdout_ratio))
    train_idx = df_sorted.index[:cutoff]
    test_idx = df_sorted.index[cutoff:]
    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def _build_lr_pipeline() -> Pipeline:
    """Build the LR preprocessing + model pipeline."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", numeric_pipe, NUMERIC_COLS),
        ],
    )
    model = LogisticRegression(max_iter=5000, solver="lbfgs", C=0.05)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_baseline(engine, model_dir: Path) -> TrainResult:
    print("[baseline] Loading training data...", flush=True)
    df = load_training_data(engine)
    if df.empty:
        raise RuntimeError(
            "No training data available. Did you load NBA stats and build features?"
        )
    print(f"[baseline] Loaded rows: {len(df)}", flush=True)

    print("[baseline] Preparing features...", flush=True)
    X, y, df_used = prepare_lr_features(df)
    X, y, df_used = (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        df_used.reset_index(drop=True),
    )
    if df_used.empty:
        raise RuntimeError(
            "Not enough training data after cleaning. Did you load NBA stats?"
        )
    if y.nunique() < 2:
        raise RuntimeError("Not enough class variety to train yet.")
    if len(df_used) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"Not enough training data available yet (rows={len(df_used)})."
        )

    folds = time_series_cv_split(df_used, X, y, n_splits=5)
    print(f"[baseline] Time-series folds: {len(folds)}", flush=True)

    oof_proba = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), np.nan)
    fold_metrics: list[dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*encountered in matmul", category=RuntimeWarning
        )
        for i, (X_tr, X_te, y_tr, y_te) in enumerate(folds):
            print(
                "[baseline] CV fold "
                f"{i + 1}/{len(folds)} train={len(X_tr)} test={len(X_te)}",
                flush=True,
            )
            pipe = _build_lr_pipeline()
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = pipe.predict(X_te)

            oof_proba[X_te.index] = proba
            oof_pred[X_te.index] = pred

            acc = float(accuracy_score(y_te, pred))
            auc = (
                float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else None
            )
            fold_metrics.append({"fold": i, "accuracy": acc, "roc_auc": auc})

    # OOF metrics (only on rows that were in test folds)
    oof_mask = ~np.isnan(oof_proba)
    oof_y = y.values[oof_mask]
    oof_p = oof_proba[oof_mask]
    oof_d = oof_pred[oof_mask]

    # Calibration on OOF predictions (more data = better calibration)
    conformal = ConformalCalibrator.calibrate(oof_p, oof_y, alpha=0.10)
    calibrator_data = None
    try:
        calibrator = best_calibrator(oof_p, oof_y)
        calibrator_data = calibrator.to_dict()
    except ValueError:
        pass

    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    auc_vals = [m["roc_auc"] for m in fold_metrics if m["roc_auc"] is not None]
    mean_auc = float(np.mean(auc_vals)) if auc_vals else None

    metrics: dict[str, Any] = {
        "accuracy": mean_acc,
        "roc_auc": mean_auc,
        "oof_accuracy": float(accuracy_score(oof_y, oof_d)),
        "oof_roc_auc": (
            float(roc_auc_score(oof_y, oof_p)) if len(np.unique(oof_y)) > 1 else None
        ),
        "n_folds": len(folds),
        "conformal_q_hat": conformal.q_hat,
        "conformal_n_cal": conformal.n_cal,
    }

    # Final model: retrain on ALL data
    print("[baseline] Training final model on full dataset...", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*encountered in matmul", category=RuntimeWarning
        )
        pipeline = _build_lr_pipeline()
        pipeline.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"baseline_logreg_{timestamp}.joblib"
    artifact = {
        "model": pipeline,
        "feature_cols": {"categorical": CATEGORICAL_COLS, "numeric": NUMERIC_COLS},
        "conformal": {
            "alpha": conformal.alpha,
            "q_hat": conformal.q_hat,
            "n_cal": conformal.n_cal,
        },
    }
    if calibrator_data:
        artifact["isotonic"] = calibrator_data
    joblib.dump(artifact, model_path)

    try:
        from app.ml.artifact_store import upload_file

        upload_file(engine, model_name="baseline_logreg", file_path=model_path)
        print(f"Uploaded baseline_logreg artifact to DB ({model_path})")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: DB upload failed for baseline_logreg: {exc}")

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
                    "params": {
                        "model": "logistic_regression",
                        "max_iter": 5000,
                        "preprocess": "onehot+standardize",
                        "cv_folds": len(folds),
                    },
                    "artifact_path": str(model_path),
                }
            )
        )

    return TrainResult(
        model_path=str(model_path), metrics=metrics, rows=int(len(df_used))
    )
