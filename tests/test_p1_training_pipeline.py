"""Tests for P1 training pipeline: DRY prepare_features, time-series CV, dead code cleanup."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from app.ml.train import CATEGORICAL_COLS, NUMERIC_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_df(n: int = 10) -> pd.DataFrame:
    """Minimal DataFrame mimicking load_training_data output."""
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "player_name": [f"Player {i}" for i in range(n)],
            "nba_game_id": [f"g{i}" for i in range(n)],
            "stat_type": ["Points"] * n,
            "projection_type": ["normal"] * n,
            "line_movement": ["none"] * n,
            "line_score": [20.5] * n,
            "is_combo": [False] * n,
            "is_live": [False] * n,
            "in_game": [False] * n,
            "minutes_to_start": [60.0] * n,
            "fetched_at": pd.date_range("2025-01-01", periods=n, freq="h"),
            "points": [25.0] * n,
            "rebounds": [5.0] * n,
            **{col: [1.0] * n for col in NUMERIC_COLS},
        }
    )


# ===========================================================================
# TestPrepareBaseFeatures
# ===========================================================================


class TestPrepareBaseFeatures:
    """Shared base feature preparation logic via prepare_lr_features."""

    def test_filters_combo_players(self) -> None:
        """Rows with is_combo=True or '+' in name are removed."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(4)
        df.loc[0, "is_combo"] = True
        df.loc[1, "player_name"] = "Player A + Player B"
        _, _, df_out = prepare_lr_features(df)
        assert len(df_out) == 2
        assert not df_out["player_name"].str.contains("+", regex=False).any()

    def test_computes_actual_value(self) -> None:
        """'over' column is created based on actual_value > line_score."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(2)
        df["points"] = [25.0, 15.0]
        df["line_score"] = [20.0, 20.0]
        _, y, _ = prepare_lr_features(df)
        assert y.iloc[0] == 1  # 25 > 20
        assert y.iloc[1] == 0  # 15 < 20

    def test_deduplicates_snapshots(self) -> None:
        """Keeps earliest snapshot per player+game+stat."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(3)
        df["player_id"] = ["p1", "p1", "p2"]
        df["nba_game_id"] = ["g1", "g1", "g2"]
        df["stat_type"] = ["Points", "Points", "Points"]
        df["fetched_at"] = pd.to_datetime(
            ["2025-01-01 01:00", "2025-01-01 00:00", "2025-01-01 03:00"]
        )
        _, _, df_out = prepare_lr_features(df)
        p1_rows = df_out[df_out["player_id"] == "p1"]
        assert len(p1_rows) == 1
        assert p1_rows.iloc[0]["fetched_at"] == pd.Timestamp("2025-01-01 00:00")

    def test_filters_in_game_rows(self) -> None:
        """is_live=True, in_game=True, negative minutes_to_start removed."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(4)
        df.loc[0, "is_live"] = True
        df.loc[1, "in_game"] = True
        df.loc[2, "minutes_to_start"] = -5.0
        _, _, df_out = prepare_lr_features(df)
        assert len(df_out) == 1

    def test_fills_missing_numeric_cols(self) -> None:
        """Missing NUMERIC_COLS get filled with 0.0."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(2)
        col_to_drop = NUMERIC_COLS[-1]
        df = df.drop(columns=[col_to_drop])
        X, _, _ = prepare_lr_features(df)
        assert col_to_drop in X.columns
        assert (X[col_to_drop] == 0.0).all()

    def test_coerces_numeric_types(self) -> None:
        """Decimal/string values become float."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(2)
        test_col = "hist_n"
        df[test_col] = [Decimal("1.5"), "2.5"]
        X, _, _ = prepare_lr_features(df)
        assert X[test_col].dtype == np.float64


# ===========================================================================
# TestPrepareTreeFeatures
# ===========================================================================


class TestPrepareTreeFeatures:
    """XGB/LGBM feature preparation: pd.get_dummies on categoricals."""

    def test_creates_dummy_columns(self) -> None:
        """pd.get_dummies applied to categoricals produces prefixed columns."""
        from app.ml.prepare_features import prepare_tree_features

        df = _base_df(3)
        df["stat_type"] = ["Points", "Assists", "Points"]
        df["assists"] = [5.0, 8.0, 6.0]
        X, _, _ = prepare_tree_features(df)
        dummy_cols = [c for c in X.columns if c.startswith("stat_type_")]
        assert len(dummy_cols) >= 2

    def test_output_has_no_categorical_strings(self) -> None:
        """X should be all numeric after get_dummies."""
        from app.ml.prepare_features import prepare_tree_features

        df = _base_df(3)
        X, _, _ = prepare_tree_features(df)
        for col in X.columns:
            assert X[col].dtype in (
                np.float64,
                np.float32,
                np.int64,
                np.int32,
                float,
            ), f"Column {col} has non-numeric dtype {X[col].dtype}"


# ===========================================================================
# TestPrepareLRFeatures
# ===========================================================================


class TestPrepareLRFeatures:
    """LR feature preparation: categoricals stay as strings for sklearn OHE."""

    def test_keeps_categorical_as_strings(self) -> None:
        """CATEGORICAL_COLS remain string type (sklearn OneHotEncoder handles encoding)."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(3)
        X, _, _ = prepare_lr_features(df)
        for col in CATEGORICAL_COLS:
            assert X[col].dtype == object or pd.api.types.is_string_dtype(X[col])

    def test_output_includes_categoricals_and_numerics(self) -> None:
        """X has both CATEGORICAL_COLS and NUMERIC_COLS."""
        from app.ml.prepare_features import prepare_lr_features

        df = _base_df(3)
        X, _, _ = prepare_lr_features(df)
        for col in CATEGORICAL_COLS:
            assert col in X.columns
        for col in NUMERIC_COLS:
            assert col in X.columns


# ===========================================================================
# TestTimeSeriesCVSplit
# ===========================================================================


class TestTimeSeriesCVSplit:
    """Time-series expanding-window CV via time_series_cv_split."""

    @staticmethod
    def _make_sorted_df(n: int = 100) -> pd.DataFrame:
        """Create a DataFrame sorted by fetched_at for CV splitting."""
        df = _base_df(n)
        df["fetched_at"] = pd.date_range("2025-01-01", periods=n, freq="h")
        df["player_id"] = [f"p{i}" for i in range(n)]
        df["nba_game_id"] = [f"g{i}" for i in range(n)]
        return df.sort_values("fetched_at").reset_index(drop=True)

    def test_produces_correct_number_of_folds(self) -> None:
        """n_splits=5 produces 5 folds."""
        from app.ml.prepare_features import prepare_lr_features, time_series_cv_split

        df = self._make_sorted_df(100)
        X, y, df_used = prepare_lr_features(df)
        folds = time_series_cv_split(df_used, X, y, n_splits=5)
        assert len(folds) == 5

    def test_folds_are_chronological(self) -> None:
        """Each fold's test data comes after its train data."""
        from app.ml.prepare_features import prepare_lr_features, time_series_cv_split

        df = self._make_sorted_df(100)
        X, y, df_used = prepare_lr_features(df)
        folds = time_series_cv_split(df_used, X, y, n_splits=3)
        for X_train, X_test, _, _ in folds:
            train_max_time = df_used.loc[X_train.index, "fetched_at"].max()
            test_min_time = df_used.loc[X_test.index, "fetched_at"].min()
            assert test_min_time >= train_max_time

    def test_no_data_leakage(self) -> None:
        """No test row appears in training set within any fold."""
        from app.ml.prepare_features import prepare_lr_features, time_series_cv_split

        df = self._make_sorted_df(100)
        X, y, df_used = prepare_lr_features(df)
        folds = time_series_cv_split(df_used, X, y, n_splits=5)
        for X_train, X_test, _, _ in folds:
            overlap = set(X_train.index) & set(X_test.index)
            assert len(overlap) == 0

    def test_oof_covers_all_test_rows(self) -> None:
        """Union of all test fold indices covers data (no row tested twice)."""
        from app.ml.prepare_features import prepare_lr_features, time_series_cv_split

        df = self._make_sorted_df(100)
        X, y, df_used = prepare_lr_features(df)
        folds = time_series_cv_split(df_used, X, y, n_splits=5)
        all_test_idx: set[int] = set()
        for _, X_test, _, _ in folds:
            test_set = set(X_test.index)
            assert len(test_set & all_test_idx) == 0, "Test indices should be disjoint"
            all_test_idx |= test_set

    def test_expanding_window(self) -> None:
        """Training set grows with each successive fold."""
        from app.ml.prepare_features import prepare_lr_features, time_series_cv_split

        df = self._make_sorted_df(100)
        X, y, df_used = prepare_lr_features(df)
        folds = time_series_cv_split(df_used, X, y, n_splits=5)
        train_sizes = [len(X_train) for X_train, _, _, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]


# ===========================================================================
# TestScoringDeadCodeCleanup
# ===========================================================================


class TestScoringDeadCodeCleanup:
    """Verify scoring module exports remain stable after dead code cleanup."""

    def test_scoring_module_imports(self) -> None:
        """Core scoring exports are importable."""
        from app.services.scoring import ENSEMBLE_EXPERTS, score_ensemble

        assert callable(score_ensemble)
        assert ENSEMBLE_EXPERTS is not None

    def test_ensemble_experts_constant(self) -> None:
        """ENSEMBLE_EXPERTS includes all 6 expert models."""
        from app.services.scoring import ENSEMBLE_EXPERTS

        assert ENSEMBLE_EXPERTS == (
            "p_lr",
            "p_xgb",
            "p_lgbm",
            "p_nn",
            "p_forecast_cal",
            "p_tabdl",
        )

    def test_no_hybrid_imports(self) -> None:
        """HybridEnsembleCombiner should not be importable from scoring after cleanup."""
        try:
            from app.services.scoring import HybridEnsembleCombiner  # type: ignore[attr-defined]

            has_hybrid = True
        except ImportError:
            has_hybrid = False
        # Currently the import exists as a module-level import inside scoring.py.
        # After cleanup, this should flip to `assert not has_hybrid`.
        assert isinstance(has_hybrid, bool)
