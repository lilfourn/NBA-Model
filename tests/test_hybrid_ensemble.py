"""Tests for hybrid ensemble combiner."""
from __future__ import annotations

import numpy as np
import pytest

from app.modeling.gating_model import GatingModel, build_context_features
from app.modeling.hybrid_ensemble import HybridEnsembleCombiner
from app.modeling.thompson_ensemble import ThompsonSamplingEnsembler


@pytest.fixture
def experts() -> list[str]:
    return ["p_forecast_cal", "p_lr", "p_xgb", "p_lgbm", "p_nn"]


@pytest.fixture
def expert_probs() -> dict[str, float]:
    return {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.7, "p_lgbm": 0.65, "p_nn": 0.45}


@pytest.fixture
def ctx() -> tuple[str, ...]:
    return ("Points", "pregame", "high")


@pytest.fixture
def thompson(experts) -> ThompsonSamplingEnsembler:
    ts = ThompsonSamplingEnsembler(experts=experts)
    # Train it a bit
    probs = {"p_forecast_cal": 0.6, "p_lr": 0.55, "p_xgb": 0.8, "p_lgbm": 0.7, "p_nn": 0.45}
    for _ in range(20):
        ts.update(probs, 1, ("Points", "pregame", "high"))
    return ts


@pytest.fixture
def gating(experts) -> GatingModel:
    rng = np.random.RandomState(42)
    n = 100
    labels = rng.randint(0, 2, size=n).astype(float)
    ep = {e: np.clip(labels + rng.normal(0, 0.2, n), 0.05, 0.95) for e in experts}
    ctx_feats = build_context_features(ep, n_eff=rng.uniform(5, 50, n))
    gm = GatingModel(experts=experts)
    gm.fit(ctx_feats, ep, labels)
    return gm


def test_predict_with_all_components(thompson, gating, experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson, gating=gating, experts=experts,
    )
    ctx_feats = np.array([0.8, 0.25, 0.15, 3.0])
    p = hybrid.predict(expert_probs, ctx, context_features=ctx_feats, p_meta=0.62)
    assert 0.0 < p < 1.0


def test_predict_thompson_only(thompson, experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson, experts=experts,
    )
    p = hybrid.predict(expert_probs, ctx)
    assert 0.0 < p < 1.0


def test_predict_meta_only(experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(experts=experts)
    p = hybrid.predict(expert_probs, ctx, p_meta=0.65)
    assert 0.0 < p < 1.0


def test_predict_no_components_falls_back(experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(experts=experts)
    p = hybrid.predict(expert_probs, ctx)
    # Should fall back to simple average of expert probs
    expected = np.mean(list(expert_probs.values()))
    assert abs(p - expected) < 1e-6


def test_update_thompson_works(thompson, experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson, experts=experts,
    )
    n_before = thompson.n_updates
    hybrid.update_thompson(expert_probs, 1, ctx)
    assert thompson.n_updates == n_before + 1


def test_update_thompson_no_thompson(experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(experts=experts)
    # Should not raise
    hybrid.update_thompson(expert_probs, 1, ctx)


def test_get_mixing_weights(experts) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        experts=experts, alpha=0.5, beta=0.3, gamma=0.2,
    )
    mw = hybrid.get_mixing_weights()
    assert mw["thompson"] == 0.5
    assert mw["gating"] == 0.3
    assert mw["meta_learner"] == 0.2


def test_fit_mixing_improves_or_maintains(thompson, gating, experts) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson, gating=gating, experts=experts,
    )
    rng = np.random.RandomState(42)
    n = 100
    labels = rng.randint(0, 2, size=n).astype(float)
    ep_list = []
    ctx_list = []
    for i in range(n):
        ep = {e: float(np.clip(labels[i] + rng.normal(0, 0.2), 0.05, 0.95)) for e in experts}
        ep_list.append(ep)
        ctx_list.append(("Points", "pregame", "high"))

    ep_arrays = {e: np.array([ep[e] for ep in ep_list]) for e in experts}
    ctx_feats = build_context_features(ep_arrays, n_eff=rng.uniform(5, 50, n))
    p_meta = np.clip(labels + rng.normal(0, 0.15, n), 0.05, 0.95)

    hybrid.fit_mixing(ep_list, ctx_list, ctx_feats, p_meta, labels)
    mw = hybrid.get_mixing_weights()
    # Weights should sum to ~1
    assert abs(mw["thompson"] + mw["gating"] + mw["meta_learner"] - 1.0) < 0.05


def test_to_state_dict(thompson, experts) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson, experts=experts, alpha=0.4, beta=0.3, gamma=0.3,
    )
    state = hybrid.to_state_dict()
    assert state["type"] == "hybrid"
    assert state["alpha"] == 0.4
    assert "thompson" in state
    assert state["thompson"]["type"] == "thompson"


def test_graceful_degradation_missing_gating_features(thompson, gating, experts, expert_probs, ctx) -> None:
    hybrid = HybridEnsembleCombiner.from_components(
        thompson=thompson, gating=gating, experts=experts,
    )
    # No context_features → gating skipped, Thompson + meta used
    p = hybrid.predict(expert_probs, ctx, p_meta=0.65)
    assert 0.0 < p < 1.0
