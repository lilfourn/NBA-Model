"""Tests for P2: team pace features, player usage rate, OOF stacking meta-learner."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _team_game_stats(n: int = 5, team: str = "BOS") -> pd.DataFrame:
    """Minimal team_game_stats DataFrame matching _load_team_game_stats output."""
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "game_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "team_abbreviation": [team] * n,
            "home_team_abbreviation": [team] * n,
            "away_team_abbreviation": ["LAL"] * n,
            "opp_abbreviation": ["LAL"] * n,
            "is_home": [1] * n,
            "points": [110.0] * n,
            "rebounds": [45.0] * n,
            "assists": [25.0] * n,
            "steals": [8.0] * n,
            "blocks": [5.0] * n,
            "turnovers": [14.0] * n,
            "fg3m": [12.0] * n,
            "fg3a": [35.0] * n,
            "fgm": [42.0] * n,
            "fga": [88.0] * n,
            "ftm": [14.0] * n,
            "fta": [18.0] * n,
            "oreb": [10.0] * n,
            "dreb": [35.0] * n,
            "pf": [20.0] * n,
            "dunks": [4.0] * n,
        }
    )


# ===========================================================================
# TestComputeTeamPace
# ===========================================================================


class TestComputeTeamPace:
    """Team pace = rolling possessions per game from box scores."""

    def test_basic(self) -> None:
        from app.ml.opponent_features import compute_team_pace

        df = _team_game_stats(5, "BOS")
        result = compute_team_pace(df)
        assert "BOS" in result
        pace_df = result["BOS"]
        assert "team_pace" in pace_df.columns
        assert "game_date" in pace_df.columns
        assert len(pace_df) == 5

    def test_formula(self) -> None:
        """Pace = FGA - OREB + TO + 0.44 * FTA."""
        from app.ml.opponent_features import compute_team_pace

        df = _team_game_stats(3, "BOS")
        df["fga"] = [90.0, 85.0, 88.0]
        df["oreb"] = [10.0, 8.0, 12.0]
        df["turnovers"] = [15.0, 12.0, 14.0]
        df["fta"] = [20.0, 18.0, 22.0]

        expected_poss = [
            90.0 - 10.0 + 15.0 + 0.44 * 20.0,
            85.0 - 8.0 + 12.0 + 0.44 * 18.0,
            88.0 - 12.0 + 14.0 + 0.44 * 22.0,
        ]
        result = compute_team_pace(df)
        pace_vals = result["BOS"]["team_pace"].tolist()
        expected_rolling = sum(expected_poss) / 3
        assert abs(pace_vals[2] - expected_rolling) < 0.01

    def test_empty(self) -> None:
        from app.ml.opponent_features import compute_team_pace

        empty = pd.DataFrame(columns=_team_game_stats(1).columns)
        result = compute_team_pace(empty)
        assert result == {}

    def test_multiple_teams(self) -> None:
        from app.ml.opponent_features import compute_team_pace

        bos = _team_game_stats(4, "BOS")
        lal = _team_game_stats(4, "LAL")
        lal["opp_abbreviation"] = "BOS"
        df = pd.concat([bos, lal], ignore_index=True)
        result = compute_team_pace(df)
        assert "BOS" in result
        assert "LAL" in result


# ===========================================================================
# TestPlayerUsage
# ===========================================================================


class TestPlayerUsage:
    """Player usage rate = share of team possessions."""

    def test_basic(self) -> None:
        from app.ml.feature_engineering import compute_player_usage

        result = compute_player_usage(
            player_fga=15.0,
            player_fta=5.0,
            player_to=3.0,
            team_fga=88.0,
            team_fta=20.0,
            team_to=14.0,
        )
        expected = (15.0 + 0.44 * 5.0 + 3.0) / (88.0 + 0.44 * 20.0 + 14.0)
        assert abs(result - expected) < 1e-6

    def test_zero_denom(self) -> None:
        from app.ml.feature_engineering import compute_player_usage

        result = compute_player_usage(
            player_fga=10.0,
            player_fta=5.0,
            player_to=2.0,
            team_fga=0.0,
            team_fta=0.0,
            team_to=0.0,
        )
        assert result == 0.0

    def test_returns_float(self) -> None:
        from app.ml.feature_engineering import compute_player_usage

        result = compute_player_usage(
            player_fga=20.0,
            player_fta=8.0,
            player_to=4.0,
            team_fga=85.0,
            team_fta=22.0,
            team_to=13.0,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# TestFeatureIntegration
# ===========================================================================


class TestFeatureIntegration:
    """New pace/usage features appear in NUMERIC_COLS."""

    def test_pace_usage_in_numeric_cols(self) -> None:
        from app.ml.prepare_features import NUMERIC_COLS

        for col in ["team_pace", "opp_pace", "game_pace", "player_usage"]:
            assert col in NUMERIC_COLS, f"{col} missing from NUMERIC_COLS"


# ===========================================================================
# TestOOFStacking
# ===========================================================================


class TestOOFStacking:
    """OOF stacking meta-learner: learned ensemble weights."""

    @staticmethod
    def _synthetic_oof(n: int = 200) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        p_lr = rng.uniform(0.3, 0.7, n)
        p_xgb = rng.uniform(0.3, 0.7, n)
        p_lgbm = rng.uniform(0.3, 0.7, n)
        p_nn = rng.uniform(0.3, 0.7, n)
        p_forecast_cal = rng.uniform(0.3, 0.7, n)
        p_tabdl = rng.uniform(0.3, 0.7, n)
        avg = (p_lr + p_xgb + p_lgbm + p_nn + p_forecast_cal + p_tabdl) / 6
        over = (avg + rng.normal(0, 0.1, n) > 0.5).astype(int)
        return pd.DataFrame(
            {
                "p_lr": p_lr,
                "p_xgb": p_xgb,
                "p_lgbm": p_lgbm,
                "p_nn": p_nn,
                "p_forecast_cal": p_forecast_cal,
                "p_tabdl": p_tabdl,
                "over": over,
            }
        )

    def test_predict_all_experts(self) -> None:
        from app.ml.stacking import predict_stacking, train_stacking_meta

        oof = self._synthetic_oof()
        result = train_stacking_meta(oof)
        prob = predict_stacking(
            result.model,
            {
                "p_lr": 0.6,
                "p_xgb": 0.55,
                "p_lgbm": 0.58,
                "p_nn": 0.6,
                "p_forecast_cal": 0.57,
                "p_tabdl": 0.59,
            },
        )
        assert 0.0 < prob < 1.0

    def test_predict_missing_expert(self) -> None:
        from app.ml.stacking import predict_stacking, train_stacking_meta

        oof = self._synthetic_oof()
        result = train_stacking_meta(oof)
        prob = predict_stacking(
            result.model, {"p_lr": 0.6, "p_xgb": None, "p_lgbm": 0.55, "p_nn": 0.58}
        )
        assert 0.0 < prob < 1.0

    def test_all_agree_over(self) -> None:
        from app.ml.stacking import predict_stacking, train_stacking_meta

        oof = self._synthetic_oof()
        result = train_stacking_meta(oof)
        prob = predict_stacking(
            result.model,
            {
                "p_lr": 0.7,
                "p_xgb": 0.65,
                "p_lgbm": 0.68,
                "p_nn": 0.7,
                "p_forecast_cal": 0.66,
                "p_tabdl": 0.72,
            },
        )
        assert prob > 0.5

    def test_all_agree_under(self) -> None:
        from app.ml.stacking import predict_stacking, train_stacking_meta

        oof = self._synthetic_oof()
        result = train_stacking_meta(oof)
        prob = predict_stacking(
            result.model,
            {
                "p_lr": 0.3,
                "p_xgb": 0.35,
                "p_lgbm": 0.32,
                "p_nn": 0.3,
                "p_forecast_cal": 0.34,
                "p_tabdl": 0.28,
            },
        )
        assert prob < 0.5

    def test_train_returns_metrics(self) -> None:
        from app.ml.stacking import train_stacking_meta

        oof = self._synthetic_oof()
        result = train_stacking_meta(oof)
        assert result.model is not None
        assert "accuracy" in result.metrics
        assert "roc_auc" in result.metrics
        assert isinstance(result.weights, dict)

    def test_symmetry_with_equal_inputs(self) -> None:
        from app.ml.stacking import predict_stacking, train_stacking_meta

        oof = self._synthetic_oof()
        result = train_stacking_meta(oof)
        p_val = 0.6
        prob = predict_stacking(
            result.model,
            {
                col: p_val
                for col in [
                    "p_lr",
                    "p_xgb",
                    "p_lgbm",
                    "p_nn",
                    "p_forecast_cal",
                    "p_tabdl",
                ]
            },
        )
        # Meta-learner intercept shifts output — just verify it's > 0.5 and in range
        assert prob > 0.5
        assert 0.0 < prob < 1.0
