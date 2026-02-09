"""Shared feature preparation and column constants for all ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.stat_mappings import stat_value_from_row

CATEGORICAL_COLS = ["stat_type", "projection_type", "line_movement"]
NUMERIC_COLS = [
    "line_score",
    "line_score_delta",
    "minutes_to_start",
    "hist_n",
    "hist_std",
    "p_hist_over",
    "rest_days",
    "is_back_to_back",
    "stat_mean_10",
    "minutes_mean_10",
    "is_home",
    "opp_def_stat_avg",
    "opp_def_points_avg",
    "opp_def_rebounds_avg",
    "opp_def_assists_avg",
    "trend_slope",
    "stat_cv",
    "recent_vs_season",
    "minutes_trend",
    "stat_std_5",
    "line_move_pct",
    "line_move_late",
    "opp_def_rank",
    "z_line",
    "mu_stab",
    "hot_streak_count",
    "cold_streak_count",
    "line_vs_opp_def",
    "forecast_edge",
    "opp_def_ratio",
    "team_pace",
    "opp_pace",
    "game_pace",
    "player_usage",
]


def prepare_base_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Common feature prep: filter combos, compute target, dedup, coerce types."""
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

    dedup_cols = ["player_id", "nba_game_id", "stat_type"]
    if all(c in df.columns for c in dedup_cols):
        df = df.sort_values("fetched_at").drop_duplicates(
            subset=dedup_cols, keep="first"
        )

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown").astype(str)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0.0).astype(float)
    if "trending_count" in df.columns:
        df["trending_count"] = np.log1p(df["trending_count"].clip(lower=0.0))

    y = df["over"]
    return df, y


def time_series_cv_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Expanding-window time-series CV splits sorted by fetched_at.

    Returns list of (X_train, X_test, y_train, y_test) tuples.
    """
    from sklearn.model_selection import TimeSeriesSplit

    df_sorted = df.sort_values("fetched_at")
    sorted_idx = df_sorted.index

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_pos, test_pos in tscv.split(sorted_idx):
        train_idx = sorted_idx[train_pos]
        test_idx = sorted_idx[test_pos]
        folds.append(
            (
                X.loc[train_idx],
                X.loc[test_idx],
                y.loc[train_idx],
                y.loc[test_idx],
            )
        )
    return folds


def prepare_tree_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare features for tree-based models (XGB, LGBM). One-hot encodes categoricals."""
    df_clean, y = prepare_base_features(df)
    cat_dummies = pd.get_dummies(
        df_clean[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, dtype=float
    )
    X = pd.concat([cat_dummies, df_clean[NUMERIC_COLS]], axis=1)
    return X, y, df_clean


def prepare_lr_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare features for LR model. Keeps categoricals as strings for sklearn pipeline."""
    df_clean, y = prepare_base_features(df)
    X = df_clean[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    return X, y, df_clean
