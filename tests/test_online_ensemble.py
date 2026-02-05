from __future__ import annotations

import math

from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler


def test_predict_fallback_to_half_when_no_experts() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"])
    ctx = Context(stat_type="Points", is_live=False, n_eff=10.0)
    assert ens.predict({}, ctx) == 0.5
    assert ens.predict({"a": None, "b": None}, ctx) == 0.5


def test_predict_averages_logits_not_probs() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"], eta=0.2, shrink_to_uniform=0.0)
    ctx = Context(stat_type="Points", is_live=False, n_eff=10.0)
    # logit(0.8) == -logit(0.2), so uniform-logit-avg => 0.5
    p = ens.predict({"a": 0.8, "b": 0.2}, ctx)
    assert abs(p - 0.5) < 1e-9


def test_update_shifts_weight_toward_better_expert() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"], eta=1.0, shrink_to_uniform=0.0)
    ctx = Context(stat_type="Assists", is_live=False, n_eff=10.0)

    ens.update({"a": 0.9, "b": 0.1}, y=1, ctx=ctx)
    w = ens.weights_for_context(ctx)
    assert w["a"] > w["b"]

    ens2 = ContextualHedgeEnsembler(experts=["a", "b"], eta=1.0, shrink_to_uniform=0.0)
    ens2.update({"a": 0.9, "b": 0.1}, y=0, ctx=ctx)
    w2 = ens2.weights_for_context(ctx)
    assert w2["b"] > w2["a"]


def test_save_and_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "ensemble_weights.json"
    ens = ContextualHedgeEnsembler(experts=["p_forecast_cal", "p_nn", "p_lr"], eta=0.5, shrink_to_uniform=0.0)
    ctx = Context(stat_type="Points", is_live=False, n_eff=20.0)
    ens.update({"p_forecast_cal": 0.7, "p_nn": 0.6, "p_lr": 0.4}, y=1, ctx=ctx)
    before = ens.weights_for_context(ctx)

    ens.save(path)
    loaded = ContextualHedgeEnsembler.load(path)
    after = loaded.weights_for_context(ctx)

    for key in before:
        assert abs(before[key] - after[key]) < 1e-12


def test_predict_ignores_nonfinite_expert_probs() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"])
    ctx = Context(stat_type="Points", is_live=False, n_eff=10.0)
    p = ens.predict({"a": float("nan"), "b": 0.9}, ctx)
    assert math.isfinite(p)
    assert abs(p - 0.9) < 1e-9


def test_predict_falls_back_when_context_weights_corrupt() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"])
    ctx = Context(stat_type="Points", is_live=False, n_eff=10.0)
    ens.weights[ctx.key()] = {"a": float("nan"), "b": float("nan")}
    p = ens.predict({"a": 0.8, "b": 0.2}, ctx)
    assert math.isfinite(p)
    assert abs(p - 0.5) < 1e-9


def test_update_skips_nonfinite_expert_probs() -> None:
    ens = ContextualHedgeEnsembler(experts=["a", "b"], eta=1.0, shrink_to_uniform=0.0)
    ctx = Context(stat_type="Assists", is_live=False, n_eff=10.0)
    ens.update({"a": float("nan"), "b": 0.9}, y=1, ctx=ctx)
    w = ens.weights_for_context(ctx)
    assert math.isfinite(w["a"])
    assert math.isfinite(w["b"])
    assert abs((w["a"] + w["b"]) - 1.0) < 1e-12
