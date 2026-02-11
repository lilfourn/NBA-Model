"""Tests for Thompson Sampling ensemble."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.modeling.thompson_ensemble import ThompsonSamplingEnsembler


@pytest.fixture
def experts() -> list[str]:
    return ["p_forecast_cal", "p_lr", "p_xgb", "p_lgbm", "p_nn"]


@pytest.fixture
def ts(experts: list[str]) -> ThompsonSamplingEnsembler:
    return ThompsonSamplingEnsembler(experts=experts)


def test_predict_returns_valid_probability(ts: ThompsonSamplingEnsembler) -> None:
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}
    ctx = ("Points", "pregame", "high")
    p = ts.predict(probs, ctx)
    assert 0.0 < p < 1.0


def test_predict_deterministic_is_stable(ts: ThompsonSamplingEnsembler) -> None:
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}
    ctx = ("Points", "pregame", "high")
    p1 = ts.predict(probs, ctx, deterministic=True)
    p2 = ts.predict(probs, ctx, deterministic=True)
    assert p1 == p2


def test_predict_stochastic_varies(experts: list[str]) -> None:
    ts = ThompsonSamplingEnsembler(experts=experts)
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}
    ctx = ("Points", "pregame", "high")
    # After some updates, stochastic predictions should show variance
    for _ in range(50):
        ts.update(probs, 1, ctx)
    results = [ts.predict(probs, ctx, deterministic=False) for _ in range(20)]
    # Not all identical (exploration)
    assert len(set(results)) > 1


def test_update_shifts_posteriors(ts: ThompsonSamplingEnsembler) -> None:
    ctx = ("Points", "pregame", "high")
    good_expert = {"p_xgb": 0.9, "p_lr": 0.1, "p_forecast_cal": 0.5, "p_lgbm": 0.5, "p_nn": 0.5}
    # Outcome is 1 — xgb (0.9) should get rewarded most
    for _ in range(50):
        ts.update(good_expert, 1, ctx)

    weights = ts.get_weights(ctx)
    assert weights["p_xgb"] > weights["p_lr"], "Expert closer to outcome should have higher weight"


def test_update_increments_n_updates(ts: ThompsonSamplingEnsembler) -> None:
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}
    ctx = ("Points", "pregame", "high")
    assert ts.n_updates == 0
    ts.update(probs, 1, ctx)
    assert ts.n_updates == 1
    ts.update(probs, 0, ctx)
    assert ts.n_updates == 2


def test_max_weight_cap(experts: list[str]) -> None:
    ts = ThompsonSamplingEnsembler(experts=experts, max_weight={"p_nn": 0.10})
    ctx = ("Points", "pregame", "high")
    # Even if NN gets massive alpha, cap should hold
    ts.alpha = {"[\"Points\", \"pregame\", \"high\"]": {e: 1.0 for e in experts}}
    ts.alpha["[\"Points\", \"pregame\", \"high\"]"]["p_nn"] = 1000.0
    ts.beta = {"[\"Points\", \"pregame\", \"high\"]": {e: 1.0 for e in experts}}
    weights = ts.get_weights(ctx)
    assert weights["p_nn"] <= 0.11  # small float tolerance


def test_weights_sum_to_one(ts: ThompsonSamplingEnsembler) -> None:
    ctx = ("Rebounds", "pregame", "low")
    weights = ts.get_weights(ctx)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_missing_expert_handled(ts: ThompsonSamplingEnsembler) -> None:
    probs = {"p_xgb": 0.7, "p_lgbm": 0.65}  # only 2 of 5
    ctx = ("Points", "pregame", "high")
    p = ts.predict(probs, ctx)
    assert 0.0 < p < 1.0


def test_state_dict_roundtrip(ts: ThompsonSamplingEnsembler) -> None:
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}
    ctx = ("Points", "pregame", "high")
    for _ in range(10):
        ts.update(probs, 1, ctx)

    state = ts.to_state_dict()
    restored = ThompsonSamplingEnsembler.from_state_dict(state)

    assert restored.experts == ts.experts
    assert restored.n_updates == ts.n_updates
    assert restored.alpha == ts.alpha
    assert restored.beta == ts.beta
    # Deterministic predictions should match
    p1 = ts.predict(probs, ctx, deterministic=True)
    p2 = restored.predict(probs, ctx, deterministic=True)
    assert abs(p1 - p2) < 1e-6


def test_save_load_roundtrip(ts: ThompsonSamplingEnsembler) -> None:
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}
    ctx = ("Points", "pregame", "high")
    for _ in range(5):
        ts.update(probs, 1, ctx)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "thompson_weights.json")
        ts.save(path)
        loaded = ThompsonSamplingEnsembler.load(path)
        assert loaded.n_updates == ts.n_updates
        assert loaded.experts == ts.experts
        p1 = ts.predict(probs, ctx, deterministic=True)
        p2 = loaded.predict(probs, ctx, deterministic=True)
        assert abs(p1 - p2) < 1e-6


def test_multiple_contexts_independent(ts: ThompsonSamplingEnsembler) -> None:
    ctx_a = ("Points", "pregame", "high")
    ctx_b = ("Rebounds", "pregame", "low")
    probs_a = {"p_xgb": 0.9, "p_lr": 0.1, "p_forecast_cal": 0.5, "p_lgbm": 0.5, "p_nn": 0.5}
    probs_b = {"p_xgb": 0.1, "p_lr": 0.9, "p_forecast_cal": 0.5, "p_lgbm": 0.5, "p_nn": 0.5}

    for _ in range(50):
        ts.update(probs_a, 1, ctx_a)
        ts.update(probs_b, 1, ctx_b)

    w_a = ts.get_weights(ctx_a)
    w_b = ts.get_weights(ctx_b)
    assert w_a["p_xgb"] > w_a["p_lr"]
    assert w_b["p_lr"] > w_b["p_xgb"]
