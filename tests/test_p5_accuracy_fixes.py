"""Tests for P5 accuracy improvements: combo stats, tighter clipping, shrinkage, median ES, feature pruning."""

from __future__ import annotations

import math

import numpy as np


# --- Phase 1a: Combo stats → PRIOR_ONLY ---


class TestComboStatsPriorOnly:
    COMBO_TYPES = {"Fantasy Score", "Pts+Rebs", "Pts+Asts", "Pts+Rebs+Asts"}
    ORIGINAL_TYPES = {
        "Offensive Rebounds",
        "Two Pointers Made",
        "Two Pointers Attempted",
        "Turnovers",
        "Free Throws Attempted",
        "Steals",
        "Blks+Stls",
        "Defensive Rebounds",
    }

    def test_combo_types_in_prior_only(self) -> None:
        from app.services.scoring import PRIOR_ONLY_STAT_TYPES

        for st in self.COMBO_TYPES:
            assert st in PRIOR_ONLY_STAT_TYPES, f"{st} missing from PRIOR_ONLY"

    def test_original_types_preserved(self) -> None:
        from app.services.scoring import PRIOR_ONLY_STAT_TYPES

        for st in self.ORIGINAL_TYPES:
            assert st in PRIOR_ONLY_STAT_TYPES, f"Original {st} removed from PRIOR_ONLY"

    def test_total_count(self) -> None:
        from app.services.scoring import PRIOR_ONLY_STAT_TYPES

        assert len(PRIOR_ONLY_STAT_TYPES) == 12


# --- Phase 1b: Tighter probability clipping ---


class TestTighterClipping:
    def test_scoring_floor_ceil(self) -> None:
        from app.services.scoring import _EXPERT_PROB_FLOOR, _EXPERT_PROB_CEIL

        assert _EXPERT_PROB_FLOOR == 0.25
        assert _EXPERT_PROB_CEIL == 0.75

    def test_strategies_floor_ceil(self) -> None:
        from app.ml.ensemble_strategies import EXPERT_PROB_FLOOR, EXPERT_PROB_CEIL

        assert EXPERT_PROB_FLOOR == 0.25
        assert EXPERT_PROB_CEIL == 0.75

    def test_outlier_cannot_dominate_logit_avg(self) -> None:
        """One extreme expert at 0.10 vs three moderate at 0.55 — clipping prevents domination."""
        from app.services.scoring import _clip_expert_probs, _logit

        raw = {"a": 0.10, "b": 0.55, "c": 0.55, "d": 0.55}
        clipped = _clip_expert_probs(raw)

        logits = [_logit(v) for v in clipped.values() if v is not None]
        avg_logit = sum(logits) / len(logits)
        p_final = 1.0 / (1.0 + math.exp(-avg_logit))
        assert p_final > 0.45, f"Outlier dominated: p_final={p_final}"


# --- Phase 1c: Increased shrinkage ---


class TestIncreasedShrinkage:
    def test_shrink_max_value(self) -> None:
        from app.services.scoring import SHRINK_MAX

        assert SHRINK_MAX == 0.25

    def test_low_neff_pulls_harder(self) -> None:
        from app.services.scoring import shrink_probability

        p_raw = 0.70
        p_low = shrink_probability(p_raw, n_eff=5.0)
        p_high = shrink_probability(p_raw, n_eff=50.0)
        assert p_low < p_high, f"Low n_eff should pull harder: {p_low} >= {p_high}"


# --- Phase 2a: Median early stopping ---


class TestMedianEarlyStopping:
    def test_median_of_odd_list(self) -> None:
        best_iters = [50, 80, 65, 90, 120]
        result = int(np.median(best_iters))
        assert result == 80

    def test_median_of_even_list(self) -> None:
        best_iters = [50, 80, 90, 120]
        result = int(np.median(best_iters))
        assert result == 85

    def test_empty_fallback(self) -> None:
        best_iters: list[int] = []
        result = int(np.median(best_iters)) if best_iters else None
        assert result is None


# --- Phase 2b: Feature pruning ---


class TestFeaturePruning:
    def test_removed_collinear_features(self) -> None:
        from app.ml.prepare_features import NUMERIC_COLS

        assert "stat_mean_3" not in NUMERIC_COLS
        assert "minutes_mean_3" not in NUMERIC_COLS

    def test_preserved_features(self) -> None:
        from app.ml.prepare_features import NUMERIC_COLS

        for col in [
            "stat_mean_10",
            "minutes_mean_10",
            "recent_vs_season",
            "minutes_trend",
        ]:
            assert col in NUMERIC_COLS, f"{col} should be preserved"

    def test_feature_count(self) -> None:
        from app.ml.prepare_features import NUMERIC_COLS

        assert len(NUMERIC_COLS) == 34
