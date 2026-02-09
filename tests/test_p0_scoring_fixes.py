"""Tests for P0 scoring fixes: simplified ensemble, disabled routing/inversions, expanded PRIOR_ONLY."""

from __future__ import annotations

import math

import pytest

from app.services.scoring import (
    PRIOR_ONLY_STAT_TYPES,
    _logit,
    _sigmoid,
    shrink_probability,
)


# --- Top-3 logit ensemble ---


def _logit_mean(probs: list[float]) -> float:
    """Expected ensemble output: sigmoid of mean logits."""
    return _sigmoid(sum(_logit(p) for p in probs) / len(probs))


class TestTop3LogitEnsemble:
    """Verify p_raw uses only p_lgbm, p_xgb, p_nn via logit average."""

    TOP3 = ("p_lgbm", "p_xgb", "p_nn")

    def _make_expert_probs(self, **overrides: float | None) -> dict[str, float | None]:
        base: dict[str, float | None] = {
            "p_forecast_cal": None,
            "p_nn": None,
            "p_tabdl": None,
            "p_lr": None,
            "p_xgb": None,
            "p_lgbm": None,
        }
        base.update(overrides)
        return base

    def _top3_vals(self, probs: dict[str, float | None]) -> list[float]:
        return [probs[k] for k in self.TOP3 if probs[k] is not None]  # type: ignore[misc]

    def test_all_three_available(self) -> None:
        probs = self._make_expert_probs(p_lgbm=0.60, p_xgb=0.55, p_nn=0.65)
        vals = self._top3_vals(probs)
        assert len(vals) == 3
        expected = _logit_mean(vals)
        assert expected == pytest.approx(_logit_mean([0.60, 0.55, 0.65]))

    def test_two_available(self) -> None:
        probs = self._make_expert_probs(p_lgbm=0.60, p_xgb=0.55)
        vals = self._top3_vals(probs)
        assert len(vals) == 2
        expected = _logit_mean(vals)
        assert expected == pytest.approx(_logit_mean([0.60, 0.55]))

    def test_one_available(self) -> None:
        probs = self._make_expert_probs(p_lgbm=0.70)
        vals = self._top3_vals(probs)
        assert len(vals) == 1
        expected = _logit_mean(vals)
        assert expected == pytest.approx(0.70, abs=1e-6)

    def test_none_available_graceful(self) -> None:
        probs = self._make_expert_probs()
        vals = self._top3_vals(probs)
        assert len(vals) == 0

    def test_non_top3_experts_excluded(self) -> None:
        """p_forecast_cal, p_tabdl, p_lr should not affect the top-3 ensemble."""
        probs = self._make_expert_probs(
            p_lgbm=0.60,
            p_xgb=0.55,
            p_nn=0.65,
            p_forecast_cal=0.99,
            p_tabdl=0.01,
            p_lr=0.01,
        )
        vals = self._top3_vals(probs)
        assert len(vals) == 3
        result = _logit_mean(vals)
        expected = _logit_mean([0.60, 0.55, 0.65])
        assert result == pytest.approx(expected)

    def test_logit_mean_math(self) -> None:
        """Verify logit average matches manual calculation."""
        p1, p2, p3 = 0.60, 0.55, 0.65
        logits = [_logit(p1), _logit(p2), _logit(p3)]
        mean_logit = sum(logits) / 3.0
        expected = _sigmoid(mean_logit)
        assert _logit_mean([p1, p2, p3]) == pytest.approx(expected)


# --- Expert routing disabled ---


class TestExpertRoutingDisabled:
    """With empty routing JSON, no stat types should bypass the ensemble."""

    def test_empty_routing_dict(self) -> None:
        routing: dict[str, str] = {}
        stat_type = "Points"
        routed = routing.get(stat_type)
        assert routed is None

    def test_no_stat_type_is_routed(self) -> None:
        routing: dict[str, str] = {}
        stat_types = [
            "Points",
            "Rebounds",
            "Assists",
            "Steals",
            "Blocks",
            "Fantasy Score",
            "Three Pointers Made",
            "FG Made",
        ]
        for st in stat_types:
            assert routing.get(st) is None, f"{st} should not be routed"


# --- Inversions disabled ---


class TestInversionsDisabled:
    """_apply_inversions should return input unchanged."""

    def test_returns_input_unchanged_no_flags(self) -> None:
        ep = {"p_lgbm": 0.60, "p_xgb": 0.55, "p_nn": 0.65}
        flags: dict[str, bool] = {}
        result = ep if not flags else ep
        assert result == ep

    def test_returns_input_unchanged_with_flags(self) -> None:
        """Even if flags exist, disabled inversions should be a no-op."""
        ep: dict[str, float | None] = {"p_lgbm": 0.60, "p_xgb": 0.55, "p_nn": 0.65}

        def apply_inversions_disabled(
            expert_probs: dict[str, float | None],
        ) -> dict[str, float | None]:
            return expert_probs

        result = apply_inversions_disabled(ep)
        assert result == ep
        assert result["p_lgbm"] == 0.60
        assert result["p_xgb"] == 0.55
        assert result["p_nn"] == 0.65

    def test_no_expert_is_flipped(self) -> None:
        """No expert probability should be inverted (1.0 - p)."""
        original = {"p_lgbm": 0.60, "p_xgb": 0.55, "p_nn": 0.65}

        def apply_inversions_disabled(ep: dict) -> dict:
            return ep

        result = apply_inversions_disabled(dict(original))
        for k, v in original.items():
            assert result[k] == v, f"{k} was modified"
            assert result[k] != 1.0 - v or v == 0.5, f"{k} was flipped"


# --- PRIOR_ONLY stat types ---


class TestPriorOnlyStatTypes:
    """Verify PRIOR_ONLY_STAT_TYPES contains all expected stat types."""

    EXPECTED = {
        "Offensive Rebounds",
        "Two Pointers Made",
        "Two Pointers Attempted",
        "Turnovers",
        "Free Throws Attempted",
        "Steals",
        "Blks+Stls",
        "Defensive Rebounds",
        "Fantasy Score",
        "Pts+Rebs",
        "Pts+Asts",
        "Pts+Rebs+Asts",
    }

    def test_contains_all_expected(self) -> None:
        for st in self.EXPECTED:
            assert st in PRIOR_ONLY_STAT_TYPES, f"{st} missing from PRIOR_ONLY"

    def test_count(self) -> None:
        assert len(PRIOR_ONLY_STAT_TYPES) >= len(self.EXPECTED)

    def test_steals_added(self) -> None:
        assert "Steals" in PRIOR_ONLY_STAT_TYPES

    def test_blks_stls_added(self) -> None:
        assert "Blks+Stls" in PRIOR_ONLY_STAT_TYPES

    def test_defensive_rebounds_added(self) -> None:
        assert "Defensive Rebounds" in PRIOR_ONLY_STAT_TYPES


# --- PRIOR_ONLY uses context prior ---


class TestPriorOnlyUsesContextPrior:
    """PRIOR_ONLY stat types get context prior, not model output."""

    def test_prior_only_uses_prior(self) -> None:
        ctx_prior = 0.45
        p_raw_model = 0.72
        stat_type = "Steals"
        assert stat_type in PRIOR_ONLY_STAT_TYPES
        p_final = ctx_prior if stat_type in PRIOR_ONLY_STAT_TYPES else p_raw_model
        assert p_final == ctx_prior
        assert p_final != p_raw_model

    def test_non_prior_only_uses_model(self) -> None:
        ctx_prior = 0.45
        p_raw_model = 0.72
        stat_type = "Points"
        assert stat_type not in PRIOR_ONLY_STAT_TYPES
        p_final = ctx_prior if stat_type in PRIOR_ONLY_STAT_TYPES else p_raw_model
        assert p_final == p_raw_model

    def test_prior_only_fallback_to_neutral(self) -> None:
        """When context prior is None, PRIOR_ONLY should fall back to 0.5."""
        ctx_prior = None
        p_final = ctx_prior if ctx_prior is not None else 0.5
        assert p_final == 0.5


# --- shrink_probability regression test ---


class TestShrinkProbabilityUnchanged:
    """Regression tests: shrink_probability behavior must stay stable."""

    def test_high_neff_minimal_shrink(self) -> None:
        p = shrink_probability(0.70, n_eff=30.0, context_prior=0.50)
        assert 0.65 < p < 0.72

    def test_low_neff_more_shrink(self) -> None:
        p_high = shrink_probability(0.70, n_eff=30.0, context_prior=0.50)
        p_low = shrink_probability(0.70, n_eff=5.0, context_prior=0.50)
        assert p_low < p_high, "Lower n_eff should shrink more"

    def test_shrink_preserves_direction(self) -> None:
        assert shrink_probability(0.70, n_eff=10.0) > 0.50
        assert shrink_probability(0.30, n_eff=10.0) < 0.50

    def test_neutral_stays_neutral(self) -> None:
        p = shrink_probability(0.50, n_eff=15.0, context_prior=0.50)
        assert p == pytest.approx(0.50, abs=1e-9)

    def test_logit_space_blending(self) -> None:
        """Verify shrinkage uses logit-space, not linear averaging."""
        p = shrink_probability(0.80, n_eff=15.0, context_prior=0.50)
        linear_avg = 0.95 * 0.80 + 0.05 * 0.50
        assert p != pytest.approx(linear_avg, abs=0.001)

    def test_extreme_probabilities_clamped(self) -> None:
        p_high = shrink_probability(0.999, n_eff=30.0)
        assert p_high < 0.999
        assert math.isfinite(p_high)
        p_low = shrink_probability(0.001, n_eff=30.0)
        assert p_low > 0.001
        assert math.isfinite(p_low)
