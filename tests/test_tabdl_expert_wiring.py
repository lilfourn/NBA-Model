from __future__ import annotations

from app.api.stats import EXPERT_COLS as API_EXPERT_COLS
from scripts.ml.train_online_ensemble import EXPERT_COLS_DEFAULT
from scripts.ops.log_decisions import LOG_COLS
from scripts.ops.monitor_model_health import EXPERT_COLS as HEALTH_EXPERT_COLS


def test_p_tabdl_is_in_all_expert_lists() -> None:
    assert "p_tabdl" in API_EXPERT_COLS
    assert "p_tabdl" in EXPERT_COLS_DEFAULT
    assert "p_tabdl" in HEALTH_EXPERT_COLS
    assert "p_tabdl" in LOG_COLS
