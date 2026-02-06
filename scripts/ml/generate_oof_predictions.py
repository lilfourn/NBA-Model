"""Generate out-of-fold (OOF) predictions for stacked meta-learner training.

Uses time-ordered K-fold splits. For each fold, trains LR/XGB/LGBM on the
training portion and predicts on the held-out portion. The result is a CSV
where every row has the true label + each base expert's OOF probability,
plus context columns for the upgraded meta-learner.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.engine import get_engine  # noqa: E402
from app.ml.dataset import load_training_data  # noqa: E402
from app.ml.stat_mappings import stat_value_from_row  # noqa: E402
from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS  # noqa: E402
from scripts.ml.train_baseline_model import load_env  # noqa: E402


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    if "is_combo" in df.columns:
        df = df[df["is_combo"].fillna(False) == False]  # noqa: E712
    if "player_name" in df.columns:
        df = df[
            ~df["player_name"].fillna("").astype(str).str.contains("+", regex=False)
        ]

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

    # Deduplicate: keep earliest snapshot per player+game+stat to prevent
    # the same prediction leaking across folds via multiple snapshots.
    dedup_cols = ["player_id", "nba_game_id", "stat_type"]
    if all(c in df.columns for c in dedup_cols):
        df = df.sort_values("fetched_at").drop_duplicates(
            subset=dedup_cols, keep="first"
        )

    df["_sort_ts"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    df = df.dropna(subset=["_sort_ts"]).sort_values("_sort_ts").reset_index(drop=True)

    # Derive season_id for cross-season awareness.
    # NBA seasons span Oct-Jun; games before October belong to the prior season.
    if "game_date" in df.columns:
        gd = pd.to_datetime(df["game_date"], errors="coerce")
        df["season_id"] = gd.apply(
            lambda d: f"{d.year}-{d.year + 1}"
            if pd.notna(d) and d.month >= 10
            else (f"{d.year - 1}-{d.year}" if pd.notna(d) else None)
        )
    elif "_sort_ts" in df.columns:
        df["season_id"] = df["_sort_ts"].apply(
            lambda d: f"{d.year}-{d.year + 1}"
            if pd.notna(d) and d.month >= 10
            else (f"{d.year - 1}-{d.year}" if pd.notna(d) else None)
        )

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown").astype(str)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype(float)
    if "trending_count" in df.columns:
        df["trending_count"] = np.log1p(df["trending_count"].clip(lower=0.0))

    cat_dummies = pd.get_dummies(
        df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, dtype=float
    )
    X = pd.concat([cat_dummies, df[NUMERIC_COLS]], axis=1)
    y = df["over"]
    return X, y, df


def _prequential_fold_indices(
    n_rows: int, n_folds: int
) -> list[tuple[list[int], list[int]]]:
    if n_rows < 2 or n_folds <= 0:
        return []

    fold_count = min(int(n_folds), int(n_rows))
    base_size = n_rows // fold_count
    remainder = n_rows % fold_count
    folds: list[tuple[list[int], list[int]]] = []

    cursor = 0
    for fold in range(fold_count):
        fold_size = base_size + (1 if fold < remainder else 0)
        test_start = cursor
        test_end = min(n_rows, test_start + fold_size)
        test_idx = list(range(test_start, test_end))
        train_idx = list(range(0, test_start))
        if test_idx:
            folds.append((train_idx, test_idx))
        cursor = test_end
    return folds


def _train_lr(X_train, y_train, X_test):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def _train_xgb(X_train, y_train, X_test):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_test,
                y_train.iloc[: len(X_test)] if len(X_test) <= len(y_train) else y_train,
            )
        ],
        verbose=False,
    )
    return model.predict_proba(X_test)[:, 1]


def _train_lgbm(X_train, y_train, X_test):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        num_leaves=31,
        verbosity=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def _train_nn_proxy(X_train, y_train, X_test):
    """Simple MLP as OOF proxy for the full GRU-Attention NN."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate OOF predictions for meta-learner."
    )
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--output", default="data/oof_predictions.csv")
    args = ap.parse_args()

    load_env()
    engine = get_engine(args.database_url)

    df = load_training_data(engine)
    if df.empty:
        raise SystemExit("No training data.")
    X, y, df_used = _prepare_features(df)
    n = len(df_used)
    if n < 200:
        raise SystemExit(f"Not enough data ({n} rows).")

    print(f"Generating OOF predictions: {n} rows, {args.n_folds} folds")

    # Prequential folds: train only on rows that occur before each test block.
    folds = _prequential_fold_indices(n_rows=n, n_folds=args.n_folds)
    oof_lr = np.full(n, np.nan)
    oof_xgb = np.full(n, np.nan)
    oof_lgbm = np.full(n, np.nan)
    oof_nn = np.full(n, np.nan)

    # Suppress sklearn matmul RuntimeWarnings from polynomial feature interactions
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning
    )

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        if not test_idx:
            continue

        if len(train_idx) < 50 or len(test_idx) < 10:
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        print(
            f"  Fold {fold_idx}/{len(folds)}: train={len(train_idx)} test={len(test_idx)}"
        )

        try:
            oof_lr[test_idx] = _train_lr(X_train, y_train, X_test)
        except Exception as e:
            print(f"    LR failed: {e}")

        try:
            oof_xgb[test_idx] = _train_xgb(X_train, y_train, X_test)
        except Exception as e:
            print(f"    XGB failed: {e}")

        try:
            oof_lgbm[test_idx] = _train_lgbm(X_train, y_train, X_test)
        except Exception as e:
            print(f"    LGBM failed: {e}")

        try:
            oof_nn[test_idx] = _train_nn_proxy(X_train, y_train, X_test)
        except Exception as e:
            print(f"    NN proxy failed: {e}")

    # Compute a simple forecast-like OOF: use p_hist_over from features
    # This proxies the stat forecast expert without needing full StatForecastPredictor
    oof_forecast = np.full(n, np.nan)
    if "p_hist_over" in df_used.columns:
        oof_forecast = df_used["p_hist_over"].fillna(0.5).values.astype(np.float64)
    else:
        oof_forecast[:] = 0.5

    # Context columns for meta-learner
    n_eff_vals = (
        df_used["hist_n"].values if "hist_n" in df_used.columns else np.zeros(n)
    )
    line_vs_mean = (
        df_used["line_vs_mean_ratio"].values
        if "line_vs_mean_ratio" in df_used.columns
        else np.ones(n)
    )

    result_dict = {
        "over": y.values,
        "stat_type": df_used["stat_type"].values
        if "stat_type" in df_used.columns
        else "",
        "n_eff": n_eff_vals,
        "line_vs_mean_ratio": line_vs_mean,
        "oof_forecast": oof_forecast,
        "oof_lr": oof_lr,
        "oof_xgb": oof_xgb,
        "oof_lgbm": oof_lgbm,
        "oof_nn": oof_nn,
    }
    # Include season_id for cross-season analysis
    if "season_id" in df_used.columns:
        result_dict["season_id"] = df_used["season_id"].values
    result = pd.DataFrame(result_dict)

    # Drop rows where LR/XGB/LGBM OOF is NaN (NN is optional)
    valid = result.dropna(subset=["oof_lr", "oof_xgb", "oof_lgbm"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(output, index=False)
    print(f"\nOOF predictions saved -> {output} ({len(valid)} rows)")


if __name__ == "__main__":
    main()
