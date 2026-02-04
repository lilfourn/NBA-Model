from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import pandas as pd

PRED_LOG_DEFAULT = "data/monitoring/prediction_log.csv"

LOG_COLS = [
    "created_at",
    "snapshot_id",
    "projection_id",
    "game_id",
    "player_id",
    "stat_type",
    "is_live",
    "decision_time",
    "line_score",
    "mu_hat",
    "sigma_hat",
    "p_over_raw",
    "p_over_cal",
    "p_forecast_cal",
    "p_nn",
    "p_lr",
    "p_final",
    "actual_value",
    "over_label",
    "model_version",
    "calibration_version",
    "calibration_status",
    "n_eff",
    "rank_score",
]


def _read_csv_header(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        first = handle.readline()
    if not first:
        return []
    return next(csv.reader([first.strip()]))


def _migrate_csv_schema(path: str, *, columns: list[str]) -> None:
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df.reindex(columns=columns)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_prediction_log(rows: pd.DataFrame, *, path: str = PRED_LOG_DEFAULT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    columns = LOG_COLS
    if os.path.exists(path):
        existing_cols = _read_csv_header(path)
        if existing_cols:
            columns = existing_cols + [c for c in LOG_COLS if c not in existing_cols]
            if columns != existing_cols:
                _migrate_csv_schema(path, columns=columns)

    out = rows.copy()
    now = datetime.now(timezone.utc).isoformat()
    out["created_at"] = now

    for col in columns:
        if col not in out.columns:
            out[col] = None
    out = out[columns]

    header = not os.path.exists(path)
    out.to_csv(path, mode="a", index=False, header=header, columns=columns)
