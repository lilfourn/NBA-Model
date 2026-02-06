from __future__ import annotations

from app.modeling.online_ensemble import Context, ContextualHedgeEnsembler
from app.api.stats import EXPERT_COLS as API_EXPERT_COLS
from app.services.scoring import _ensure_ensemble_experts
from scripts.ml.train_online_ensemble import EXPERT_COLS_DEFAULT
from scripts.ops.log_decisions import LOG_COLS
from scripts.ops.monitor_model_health import EXPERT_COLS as HEALTH_EXPERT_COLS


def test_p_tabdl_is_in_all_expert_lists() -> None:
    assert "p_tabdl" in API_EXPERT_COLS
    assert "p_tabdl" in EXPERT_COLS_DEFAULT
    assert "p_tabdl" in HEALTH_EXPERT_COLS
    assert "p_tabdl" in LOG_COLS


def test_ensure_ensemble_experts_adds_missing_expert_and_renormalizes() -> None:
    ens = ContextualHedgeEnsembler(experts=["p_lr", "p_xgb"], eta=0.2, shrink_to_uniform=0.0)
    ctx = Context(stat_type="PTS", is_live=False, n_eff=10.0)
    ens.update({"p_lr": 0.7, "p_xgb": 0.4}, y=1, ctx=ctx)

    ens = _ensure_ensemble_experts(ens, ["p_lr", "p_xgb", "p_tabdl"])
    weights = ens.weights_for_context(ctx)

    assert set(weights.keys()) == {"p_lr", "p_xgb", "p_tabdl"}
    assert abs(sum(weights.values()) - 1.0) < 1e-12

