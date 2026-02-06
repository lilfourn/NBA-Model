from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.ops.monitor_model_health import _check_ensemble_weights
from scripts.ops.monitor_model_health import _check_rolling_metrics
from scripts.ops.monitor_model_health import ROLLING_WINDOW


def test_check_ensemble_weights_handles_context_nested_format(tmp_path: Path) -> None:
    path = tmp_path / "ensemble_weights.json"
    payload = {
        "weights": {
            '["PTS","pregame","neff_ge15"]': {"p_nn": 0.92, "p_lr": 0.08},
            '["AST","pregame","neff_ge15"]': {"p_nn": 0.90, "p_lr": 0.10},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    alerts = _check_ensemble_weights(str(path))
    assert any(a.get("type") == "weight_collapse" and a.get("expert") == "p_nn" for a in alerts)


def test_check_ensemble_weights_handles_flat_format(tmp_path: Path) -> None:
    path = tmp_path / "ensemble_weights.json"
    payload = {"weights": {"p_nn": 0.4, "p_lr": 0.3, "p_xgb": 0.3}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    alerts = _check_ensemble_weights(str(path))
    assert alerts == []


def test_check_rolling_metrics_prefers_is_correct_for_p_final() -> None:
    rows = []
    for idx in range(ROLLING_WINDOW):
        # Deliberately make p_final threshold disagree with labels.
        p_final = 0.9 if idx % 2 == 0 else 0.1
        label = 0 if idx % 2 == 0 else 1
        rows.append(
            {
                "p_final": p_final,
                "p_forecast_cal": 0.5,
                "p_nn": 0.5,
                "p_tabdl": 0.5,
                "p_lr": 0.5,
                "p_xgb": 0.5,
                "p_lgbm": 0.5,
                "over_label": label,
                "is_correct": 1.0,  # Ground-truth stored correctness
            }
        )
    df = pd.DataFrame(rows)

    result = _check_rolling_metrics(df)
    p_final_metrics = result["metrics"]["p_final"]
    assert p_final_metrics["rolling_accuracy"] == 1.0
