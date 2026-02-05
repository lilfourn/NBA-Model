"""Tests for learned gating model."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.modeling.gating_model import GatingModel, build_context_features


@pytest.fixture
def experts() -> list[str]:
    return ["p_forecast_cal", "p_lr", "p_xgb", "p_lgbm", "p_nn"]


@pytest.fixture
def synthetic_data(experts: list[str]):
    """Synthetic dataset where XGB is best expert."""
    rng = np.random.RandomState(42)
    n = 200
    labels = rng.randint(0, 2, size=n).astype(float)
    expert_probs = {}
    for e in experts:
        noise = rng.normal(0, 0.2, size=n)
        if e == "p_xgb":
            # XGB is well-calibrated
            expert_probs[e] = np.clip(labels + rng.normal(0, 0.1, n), 0.05, 0.95)
        elif e == "p_nn":
            # NN is basically random
            expert_probs[e] = np.clip(0.5 + noise, 0.05, 0.95)
        else:
            expert_probs[e] = np.clip(labels + noise, 0.05, 0.95)
    n_eff = rng.uniform(5, 50, size=n)
    context = build_context_features(expert_probs, n_eff)
    return context, expert_probs, labels, n_eff


def test_fit_succeeds(experts, synthetic_data) -> None:
    context, expert_probs, labels, _ = synthetic_data
    gm = GatingModel(experts=experts)
    gm.fit(context, expert_probs, labels)
    assert gm.is_fitted
    assert gm.n_train == 200


def test_fit_requires_min_samples(experts) -> None:
    gm = GatingModel(experts=experts)
    with pytest.raises(ValueError, match="Need >= 30"):
        gm.fit(
            np.zeros((10, 4)),
            {e: np.zeros(10) for e in experts},
            np.zeros(10),
        )


def test_predict_weights_shape(experts, synthetic_data) -> None:
    context, expert_probs, labels, _ = synthetic_data
    gm = GatingModel(experts=experts)
    gm.fit(context, expert_probs, labels)
    weights = gm.predict_weights(context)
    assert weights.shape == (200, 5)


def test_weights_sum_to_one(experts, synthetic_data) -> None:
    context, expert_probs, labels, _ = synthetic_data
    gm = GatingModel(experts=experts)
    gm.fit(context, expert_probs, labels)
    weights = gm.predict_weights(context)
    row_sums = weights.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_predict_weights_single(experts, synthetic_data) -> None:
    context, expert_probs, labels, _ = synthetic_data
    gm = GatingModel(experts=experts)
    gm.fit(context, expert_probs, labels)
    single = context[0]
    w = gm.predict_weights_single(single)
    assert set(w.keys()) == set(experts)
    assert abs(sum(w.values()) - 1.0) < 1e-5


def test_max_weight_cap(experts, synthetic_data) -> None:
    context, expert_probs, labels, _ = synthetic_data
    gm = GatingModel(experts=experts, max_weight={"p_nn": 0.05})
    gm.fit(context, expert_probs, labels)
    weights = gm.predict_weights(context)
    nn_idx = experts.index("p_nn")
    # After softmax, the cap is applied before normalization so the
    # final weight should be reasonably small
    assert weights[:, nn_idx].max() < 0.30  # loose bound since softmax re-normalizes


def test_save_load_roundtrip(experts, synthetic_data) -> None:
    context, expert_probs, labels, _ = synthetic_data
    gm = GatingModel(experts=experts)
    gm.fit(context, expert_probs, labels)
    original_weights = gm.predict_weights(context[:5])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "gating_model.joblib")
        gm.save(path)
        loaded = GatingModel.load(path)
        loaded_weights = loaded.predict_weights(context[:5])
        np.testing.assert_allclose(original_weights, loaded_weights, atol=1e-6)
        assert loaded.is_fitted
        assert loaded.n_train == 200


def test_unfitted_model_returns_uniform(experts) -> None:
    gm = GatingModel(experts=experts)
    context = np.array([[0.8, 0.2, 0.3, 3.0]])
    weights = gm.predict_weights(context)
    # All scores default to 0.5 → uniform after softmax
    expected = 1.0 / len(experts)
    np.testing.assert_allclose(weights[0], expected, atol=1e-5)


# --- build_context_features tests ---


def test_build_context_features_shape() -> None:
    probs = {
        "p_xgb": np.array([0.7, 0.3, 0.8]),
        "p_lr": np.array([0.6, 0.4, 0.7]),
    }
    ctx = build_context_features(probs, n_eff=np.array([10, 20, 30]))
    assert ctx.shape == (3, 4)


def test_build_context_features_agreement() -> None:
    # All agree on OVER
    probs = {
        "p_a": np.array([0.8]),
        "p_b": np.array([0.9]),
    }
    ctx = build_context_features(probs)
    assert ctx[0, 0] == 1.0  # 100% agreement


def test_build_context_features_spread() -> None:
    probs = {
        "p_a": np.array([0.2]),
        "p_b": np.array([0.8]),
    }
    ctx = build_context_features(probs)
    assert abs(ctx[0, 1] - 0.6) < 1e-6  # spread = 0.8 - 0.2


def test_build_context_features_no_experts_raises() -> None:
    with pytest.raises(ValueError, match="No expert probabilities"):
        build_context_features({})
