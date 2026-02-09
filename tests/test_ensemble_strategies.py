"""Tests for ensemble_strategies: strategy correctness + shared metrics."""

from __future__ import annotations

import numpy as np

from app.ml.ensemble_strategies import (
    EnsembleStrategy,
    LogitAvgStrategy,
    PerStatStrategy,
    RecencyWeightedStrategy,
    TopKStrategy,
    build_strategies,
    multi_window_metrics,
    rolling_metrics,
)

EXPERTS = {
    "p_lr": 0.55,
    "p_xgb": 0.60,
    "p_lgbm": 0.52,
    "p_nn": 0.48,
    "p_forecast_cal": 0.45,
    "p_tabdl": 0.66,
}


# --- LogitAvgStrategy ---


class TestLogitAvgStrategy:
    def test_all_experts(self):
        s = LogitAvgStrategy()
        p = s.combine(EXPERTS)
        assert 0.0 < p < 1.0

    def test_subset_only(self):
        s = LogitAvgStrategy(experts=["p_lr", "p_xgb"])
        p = s.combine(EXPERTS)
        assert p > 0.5

    def test_missing_experts_skipped(self):
        s = LogitAvgStrategy(experts=["p_lr", "p_xgb", "p_missing"])
        probs = {"p_lr": 0.6, "p_xgb": 0.7, "p_missing": None}
        p = s.combine(probs)
        assert 0.5 < p < 1.0

    def test_all_none_returns_0_5(self):
        s = LogitAvgStrategy()
        p = s.combine({"a": None, "b": None})
        assert p == 0.5

    def test_clips_extreme_probs(self):
        s = LogitAvgStrategy()
        p = s.combine({"a": 0.01, "b": 0.99})
        assert 0.25 <= p <= 0.75


# --- RecencyWeightedStrategy ---


class TestRecencyWeightedStrategy:
    def _feed(self, s: RecencyWeightedStrategy, n: int, probs: dict, outcome: int):
        for _ in range(n):
            s.update(probs, outcome)

    def test_cold_start_uniform(self):
        s = RecencyWeightedStrategy()
        p = s.combine(EXPERTS)
        p_logit = LogitAvgStrategy().combine(EXPERTS)
        assert abs(p - p_logit) < 1e-6

    def test_accurate_expert_gets_weight(self):
        s = RecencyWeightedStrategy(window=50, temperature=5.0)
        good = {"a": 0.7, "b": 0.3}
        bad = {"a": 0.3, "b": 0.7}
        for _ in range(30):
            s.update(good, 1)
            s.update(bad, 0)

        # Expert 'a' is always right → should pull result > 0.5
        p = s.combine({"a": 0.7, "b": 0.3})
        p_uniform = LogitAvgStrategy().combine({"a": 0.7, "b": 0.3})
        assert p >= p_uniform

    def test_window_drops_old(self):
        s = RecencyWeightedStrategy(window=20)
        good = {"a": 0.8, "b": 0.2}
        for _ in range(20):
            s.update(good, 1)
        # Now feed bad data
        bad = {"a": 0.2, "b": 0.8}
        for _ in range(20):
            s.update(bad, 1)
        # Window should now only see 'b' as the good expert
        p = s.combine({"a": 0.8, "b": 0.2})
        assert p < 0.5  # 'a' is now the bad expert per recent window

    def test_all_equal_is_uniform(self):
        s = RecencyWeightedStrategy()
        probs = {"a": 0.6, "b": 0.6}
        for _ in range(20):
            s.update(probs, 1)
        p = s.combine(probs)
        p_logit = LogitAvgStrategy().combine(probs)
        assert abs(p - p_logit) < 0.02


# --- PerStatStrategy ---


class TestPerStatStrategy:
    def test_cold_start_fallback(self):
        s = PerStatStrategy(min_history=20)
        s._current_stat_type = "Points"
        p = s.combine(EXPERTS)
        p_logit = LogitAvgStrategy().combine(EXPERTS)
        assert abs(p - p_logit) < 1e-6

    def test_routes_to_best_per_stat(self):
        s = PerStatStrategy(window=100, min_history=10)
        # Feed: expert 'a' always right for Points
        for _ in range(20):
            s.update({"a": 0.7, "b": 0.3}, outcome=1, stat_type="Points")
            s.update({"a": 0.3, "b": 0.7}, outcome=0, stat_type="Points")
        # Expert 'a' should be best for Points
        s._current_stat_type = "Points"
        p = s.combine({"a": 0.7, "b": 0.3})
        assert p > 0.5

    def test_min_history_enforced(self):
        s = PerStatStrategy(min_history=50)
        for _ in range(10):
            s.update({"a": 0.7, "b": 0.3}, outcome=1, stat_type="Points")
        s._current_stat_type = "Points"
        p = s.combine(EXPERTS)
        p_logit = LogitAvgStrategy().combine(EXPERTS)
        assert abs(p - p_logit) < 1e-6


# --- TopKStrategy ---


class TestTopKStrategy:
    def test_cold_start_uses_all(self):
        s = TopKStrategy(k=2, min_history=20)
        p = s.combine(EXPERTS)
        p_logit = LogitAvgStrategy().combine(EXPERTS)
        assert abs(p - p_logit) < 1e-6

    def test_selects_top_k_after_warmup(self):
        s = TopKStrategy(k=1, window=100, min_history=10)
        # Expert 'a' always right, 'b' always wrong
        for _ in range(20):
            s.update({"a": 0.7, "b": 0.3}, outcome=1)
            s.update({"a": 0.3, "b": 0.7}, outcome=0)
        # k=1 should pick 'a' only
        top = s._top_k_experts()
        assert top == ["a"]

    def test_k1_picks_single_best(self):
        s = TopKStrategy(k=1, min_history=10)
        for _ in range(20):
            s.update({"a": 0.8, "b": 0.2, "c": 0.5}, outcome=1)
        p = s.combine({"a": 0.8, "b": 0.2, "c": 0.5})
        # Should use only expert 'a' (always right)
        assert p > 0.6


# --- MultiWindowMetrics ---


class TestMultiWindowMetrics:
    def test_rolling_accuracy_and_logloss(self):
        probs = np.array([0.9] * 50 + [0.1] * 50)
        labels = np.ones(100)
        m = rolling_metrics(probs, labels, window=50)
        # Last 50 are all 0.1 predicting label=1, so picks=0 → all wrong
        assert m["rolling_accuracy"] == 0.0
        assert m["rolling_brier"] > 0

    def test_insufficient_window_returns_empty(self):
        probs = np.array([0.6, 0.7])
        labels = np.array([1, 1])
        assert rolling_metrics(probs, labels, window=50) == {}

    def test_multi_window_keys(self):
        probs = np.random.default_rng(42).uniform(0.3, 0.7, 250)
        labels = np.random.default_rng(42).integers(0, 2, 250).astype(float)
        result = multi_window_metrics(probs, labels)
        assert "last_50" in result
        assert "last_100" in result
        assert "last_200" in result
        assert "all_time" in result

    def test_custom_windows(self):
        probs = np.random.default_rng(0).uniform(0.3, 0.7, 80)
        labels = np.random.default_rng(0).integers(0, 2, 80).astype(float)
        result = multi_window_metrics(probs, labels, windows=[30, 60])
        assert "last_30" in result
        assert "last_60" in result
        assert "last_50" not in result


# --- build_strategies factory ---


class TestBuildStrategies:
    def test_without_stacking(self):
        strats = build_strategies()
        assert "logit_avg" in strats
        assert "recency_weighted" in strats
        assert "per_stat" in strats
        assert "top_k" in strats
        assert "stacking" not in strats

    def test_with_stacking(self):
        class FakeModel:
            def predict_proba(self, X):
                return np.array([[0.4, 0.6]])

        strats = build_strategies(stacking_model=FakeModel())
        assert "stacking" in strats

    def test_all_share_interface(self):
        strats = build_strategies()
        for name, s in strats.items():
            assert isinstance(s, EnsembleStrategy)
            p = s.combine(EXPERTS)
            assert 0.0 < p < 1.0
