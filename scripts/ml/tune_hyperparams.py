"""Optuna hyperparameter tuning for XGBoost and LightGBM with walk-forward CV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import load_training_data  # noqa: E402
from app.ml.stat_mappings import stat_value_from_row  # noqa: E402
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def _walk_forward_cv_splits(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Time-ordered walk-forward CV splits."""
    df = df.copy()
    df["_sort_key"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    sorted_idx = df["_sort_key"].sort_values().index
    n = len(sorted_idx)
    # Reserve first 40% as initial train, split rest into n_folds test windows
    init_train_end = int(n * 0.4)
    remaining = n - init_train_end
    fold_size = remaining // n_folds

    splits = []
    for fold in range(n_folds):
        test_start = init_train_end + fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        train_idx = sorted_idx[:test_start]
        test_idx = sorted_idx[test_start:test_end]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        splits.append((
            X.loc[train_idx],
            X.loc[test_idx],
            y.loc[train_idx],
            y.loc[test_idx],
        ))
    return splits


def _objective_xgb(trial: optuna.Trial, splits: list) -> float:
    from xgboost import XGBClassifier

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": 42,
    }

    losses = []
    for X_train, X_test, y_train, y_test in splits:
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        losses.append(log_loss(y_test, y_proba))
    return float(np.mean(losses))


def _objective_lgbm(trial: optuna.Trial, splits: list) -> float:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "verbosity": -1,
        "random_state": 42,
    }

    losses = []
    for X_train, X_test, y_train, y_test in splits:
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        losses.append(log_loss(y_test, y_proba))
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna hyperparameter tuning for XGB and LightGBM.")
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--model", choices=["xgb", "lgbm", "both"], default="both")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--output-dir", default="data/tuning")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    df = load_training_data(engine)
    if df.empty:
        raise SystemExit("No training data.")
    X, y, df_used = _prepare_features(df)
    if len(df_used) < 200:
        raise SystemExit(f"Not enough data for tuning ({len(df_used)} rows).")

    splits = _walk_forward_cv_splits(df_used, X, y, n_folds=args.n_folds)
    if not splits:
        raise SystemExit("Could not create walk-forward CV splits.")
    print(f"Walk-forward CV: {len(splits)} folds, sizes: {[len(s[1]) for s in splits]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("xgb", "both"):
        print(f"\nTuning XGBoost ({args.n_trials} trials)...")
        study_xgb = optuna.create_study(direction="minimize")
        study_xgb.optimize(lambda t: _objective_xgb(t, splits), n_trials=args.n_trials)
        best_xgb = study_xgb.best_params
        print(f"XGB best logloss: {study_xgb.best_value:.4f}")
        print(f"XGB best params: {best_xgb}")
        xgb_out = output_dir / "best_params_xgb.json"
        xgb_out.write_text(json.dumps(best_xgb, indent=2), encoding="utf-8")
        print(f"Saved -> {xgb_out}")

    if args.model in ("lgbm", "both"):
        print(f"\nTuning LightGBM ({args.n_trials} trials)...")
        study_lgbm = optuna.create_study(direction="minimize")
        study_lgbm.optimize(lambda t: _objective_lgbm(t, splits), n_trials=args.n_trials)
        best_lgbm = study_lgbm.best_params
        print(f"LGBM best logloss: {study_lgbm.best_value:.4f}")
        print(f"LGBM best params: {best_lgbm}")
        lgbm_out = output_dir / "best_params_lgbm.json"
        lgbm_out.write_text(json.dumps(best_lgbm, indent=2), encoding="utf-8")
        print(f"Saved -> {lgbm_out}")


if __name__ == "__main__":
    main()
