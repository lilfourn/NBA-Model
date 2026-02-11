"""Tests for ensemble bias fixes: NN weight cap, shrinkage direction, balanced output."""

from __future__ import annotations


from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler
from app.services.scoring import shrink_probability


def test_nn_weight_cap_enforced() -> None:
    ens = ContextualHedgeEnsembler(
        experts=["a", "b", "c"],
        eta=0.2,
        max_weight={"b": 0.10},
    )
    ctx = Context(stat_type="Points", is_live=False, n_eff=10.0)
    # Force b to have high weight
    ens.weights[ctx.key()] = {"a": 0.3, "b": 0.5, "c": 0.2}
    # b should be capped at 0.10 during predict, not 0.5
    p = ens.predict({"a": 0.4, "b": 0.9, "c": 0.4}, ctx)
    # Without cap: b's 0.9 would pull strongly toward OVER
    # With cap: b's influence is limited
    # Verify the prediction is less influenced by b
    ens_no_cap = ContextualHedgeEnsembler(experts=["a", "b", "c"], eta=0.2)
    ens_no_cap.weights[ctx.key()] = {"a": 0.3, "b": 0.5, "c": 0.2}
    p_no_cap = ens_no_cap.predict({"a": 0.4, "b": 0.9, "c": 0.4}, ctx)
    assert p < p_no_cap, "NN cap should reduce influence of high-weight expert"


def test_max_weight_persisted_in_state() -> None:
    ens = ContextualHedgeEnsembler(
        experts=["a", "b"],
        max_weight={"b": 0.15},
    )
    state = ens.to_state_dict()
    assert "max_weight" in state
    assert state["max_weight"]["b"] == 0.15

    loaded = ContextualHedgeEnsembler.from_state_dict(state)
    assert loaded.max_weight == {"b": 0.15}


def test_max_weight_absent_when_empty() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"])
    state = ens.to_state_dict()
    assert "max_weight" not in state


def test_shrinkage_preserves_direction() -> None:
    """Shrinkage should pull toward anchor (0.42) but not flip far predictions."""
    # Strong OVER predictions stay above anchor
    for p_raw in [0.60, 0.70, 0.80, 0.90, 0.99]:
        p_shrunk = shrink_probability(p_raw, n_eff=None)
        assert (
            p_shrunk > 0.42
        ), f"Strong OVER p={p_raw} pulled below anchor to {p_shrunk}"
    # Strong UNDER predictions stay below anchor
    for p_raw in [0.30, 0.20, 0.10, 0.01]:
        p_shrunk = shrink_probability(p_raw, n_eff=None)
        assert (
            p_shrunk < 0.42
        ), f"Strong UNDER p={p_raw} pulled above anchor to {p_shrunk}"


def test_shrinkage_at_anchor_stays_at_anchor() -> None:
    """Shrinking the anchor value should return the anchor itself."""
    from app.services.scoring import SHRINK_ANCHOR

    p = shrink_probability(SHRINK_ANCHOR, n_eff=10.0)
    assert abs(p - SHRINK_ANCHOR) < 1e-9


def test_uniform_weights_balanced_output() -> None:
    """Ensemble with uniform weights on balanced experts should output ~0.5."""
    ens = ContextualHedgeEnsembler(experts=["a", "b", "c", "d", "e"])
    ctx = Context(stat_type="Points", is_live=False, n_eff=15.0)
    # Symmetric experts: 2 over, 2 under, 1 neutral
    p = ens.predict({"a": 0.7, "b": 0.3, "c": 0.6, "d": 0.4, "e": 0.5}, ctx)
    assert abs(p - 0.5) < 0.05, f"Balanced experts should give ~0.5, got {p}"


def test_reduced_shrinkage_values() -> None:
    """Verify shrinkage constants use base-rate anchor."""
    from app.services.scoring import SHRINK_MIN, SHRINK_MAX, SHRINK_ANCHOR

    assert SHRINK_MIN == 0.05
    assert SHRINK_MAX == 0.25
    assert SHRINK_ANCHOR == 0.50
