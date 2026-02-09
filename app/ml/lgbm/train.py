from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from app.db import schema
from app.ml.calibration import best_calibrator
from app.ml.dataset import load_training_data
from app.ml.prepare_features import prepare_tree_features, time_series_cv_split
from app.ml.prepare_features import CATEGORICAL_COLS, NUMERIC_COLS
from app.ml.train import MIN_TRAIN_ROWS
from app.modeling.conformal import ConformalCalibrator

LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "num_leaves": 10,
    "verbosity": -1,
    "random_state": 42,
}


@dataclass
class TrainResult:
    model_path: str
    metrics: dict[str, Any]
    rows: int


def _load_tuned_params() -> dict[str, Any]:
    """Load Optuna-tuned params merged with required static params."""
    params = dict(LGBM_PARAMS)
    candidates: list[Path] = []
    tuning_dir = os.getenv("TUNING_DIR", "").strip()
    if tuning_dir:
        candidates.append(Path(tuning_dir) / "best_params_lgbm.json")
    candidates.append(Path("data/tuning/best_params_lgbm.json"))
    candidates.append(Path("/state/data/tuning/best_params_lgbm.json"))

    import json

    for path in candidates:
        if not path.exists():
            continue
        try:
            tuned = json.loads(path.read_text(encoding="utf-8"))
            params.update(tuned)
            break
        except Exception:  # noqa: BLE001
            continue
    params.setdefault("verbosity", -1)
    params.setdefault("random_state", 42)
    return params


def train_lightgbm(engine, model_dir: Path) -> TrainResult:
    df = load_training_data(engine)
    if df.empty:
        raise RuntimeError("No training data available.")

    X, y, df_used = prepare_tree_features(df)
    X, y, df_used = (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        df_used.reset_index(drop=True),
    )
    if df_used.empty:
        raise RuntimeError("Not enough training data after cleaning.")
    if y.nunique() < 2:
        raise RuntimeError("Not enough class variety to train yet.")
    if len(df_used) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Not enough training data (rows={len(df_used)}).")

    import lightgbm

    params = _load_tuned_params()
    folds = time_series_cv_split(df_used, X, y, n_splits=5)

    oof_proba = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), np.nan)
    fold_metrics: list[dict[str, Any]] = []
    last_fold_model: LGBMClassifier | None = None

    for i, (X_tr, X_te, y_tr, y_te) in enumerate(folds):
        mdl = LGBMClassifier(**params)
        mdl.fit(
            X_tr,
            y_tr,
            eval_set=[(X_te, y_te)],
            callbacks=[
                lightgbm.early_stopping(50, verbose=False),
                lightgbm.log_evaluation(0),
            ],
        )
        proba = mdl.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_proba[X_te.index] = proba
        oof_pred[X_te.index] = pred

        acc = float(accuracy_score(y_te, pred))
        auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else None
        ll = float(log_loss(y_te, proba))
        fold_metrics.append({"fold": i, "accuracy": acc, "roc_auc": auc, "logloss": ll})
        last_fold_model = mdl

    oof_mask = ~np.isnan(oof_proba)
    oof_y = y.values[oof_mask]
    oof_p = oof_proba[oof_mask]
    oof_d = oof_pred[oof_mask]

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
    ll_vals = [m["logloss"] for m in fold_metrics]
    mean_ll = float(np.mean(ll_vals))

    metrics: dict[str, Any] = {
        "accuracy": mean_acc,
        "roc_auc": mean_auc,
        "logloss": mean_ll,
        "oof_accuracy": float(accuracy_score(oof_y, oof_d)),
        "oof_roc_auc": (
            float(roc_auc_score(oof_y, oof_p)) if len(np.unique(oof_y)) > 1 else None
        ),
        "n_folds": len(folds),
        "conformal_q_hat": conformal.q_hat,
        "conformal_n_cal": conformal.n_cal,
    }

    # Final model: retrain on ALL data (cap n_estimators from last fold's early stopping)
    last_best = getattr(last_fold_model, "best_iteration_", None)
    final_params = dict(params)
    if last_best:
        final_params["n_estimators"] = last_best
    final_params.pop("early_stopping_rounds", None)
    model = LGBMClassifier(**final_params)
    model.fit(X, y)
    print(f"LGBM final model trained on {len(X)} rows")

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_path = model_dir / f"lgbm_{timestamp}.joblib"
    artifact = {
        "model": model,
        "feature_cols": list(X.columns),
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
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

        upload_file(engine, model_name="lgbm", file_path=model_path)
        print(f"Uploaded lgbm artifact to DB ({model_path})")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: DB upload failed for lgbm: {exc}")

    run_id = uuid4()
    with engine.begin() as conn:
        conn.execute(
            schema.model_runs.insert().values(
                {
                    "id": run_id,
                    "created_at": datetime.now(timezone.utc),
                    "model_name": "lightgbm",
                    "train_rows": int(len(df_used)),
                    "metrics": metrics,
                    "params": params,
                    "artifact_path": str(model_path),
                }
            )
        )

    return TrainResult(
        model_path=str(model_path), metrics=metrics, rows=int(len(df_used))
    )
