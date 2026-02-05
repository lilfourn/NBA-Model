from __future__ import annotations

import pandas as pd

from scripts.ml.train_online_ensemble import _write_back_outcomes


def test_write_back_outcomes_updates_only_missing(tmp_path) -> None:
    path = tmp_path / "prediction_log.csv"
    log_df = pd.DataFrame(
        [
            {"__row_id": 0, "actual_value": None, "over_label": None, "line_score": 20.5},
            {"__row_id": 1, "actual_value": 15.0, "over_label": 0, "line_score": 18.5},
        ]
    )
    resolved = pd.DataFrame(
        [
            {"__row_id": 0, "actual_value": 22.0, "over_label": 1},
            {"__row_id": 1, "actual_value": 19.0, "over_label": 1},
        ]
    )

    updated = _write_back_outcomes(log_df, resolved, path=path, only_missing=True)
    assert updated == 1

    out = pd.read_csv(path)
    assert float(out.loc[0, "actual_value"]) == 22.0
    assert int(out.loc[0, "over_label"]) == 1
    assert float(out.loc[1, "actual_value"]) == 15.0
    assert int(out.loc[1, "over_label"]) == 0


def test_write_back_outcomes_can_overwrite_existing(tmp_path) -> None:
    path = tmp_path / "prediction_log.csv"
    log_df = pd.DataFrame(
        [
            {"__row_id": 0, "actual_value": 10.0, "over_label": 0, "line_score": 9.5},
        ]
    )
    resolved = pd.DataFrame(
        [
            {"__row_id": 0, "actual_value": 12.0, "over_label": 1},
        ]
    )

    updated = _write_back_outcomes(log_df, resolved, path=path, only_missing=False)
    assert updated == 1

    out = pd.read_csv(path)
    assert float(out.loc[0, "actual_value"]) == 12.0
    assert int(out.loc[0, "over_label"]) == 1
