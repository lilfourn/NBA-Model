from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.modeling.stabilization import STABILIZATION_GAMES
from app.ml.stat_mappings import SPECIAL_DIFFS, STAT_COMPONENTS, stat_components, stat_diff_components


def prepare_gamelogs(frame: pd.DataFrame) -> pd.DataFrame:
    logs = frame.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    return logs.sort_values(["player_id", "game_date"])


def compute_league_means(gamelogs: pd.DataFrame) -> dict[str, float]:
    means: dict[str, float] = {}
    for stat_key, cols in STAT_COMPONENTS.items():
        if not cols:
            continue
        if any(col not in gamelogs.columns for col in cols):
            continue
        series = gamelogs[cols].fillna(0).sum(axis=1)
        means[stat_key] = float(series.mean()) if len(series) else 0.0

    for stat_key, (base_col, sub_col) in SPECIAL_DIFFS.items():
        if base_col not in gamelogs.columns or sub_col not in gamelogs.columns:
            continue
        series = gamelogs[base_col].fillna(0) - gamelogs[sub_col].fillna(0)
        means[stat_key] = float(series.mean()) if len(series) else 0.0
    return means


def _cutoff_date(cutoff: datetime | None) -> datetime | None:
    if cutoff is None:
        return None
    cutoff_ts = cutoff if isinstance(cutoff, pd.Timestamp) else pd.to_datetime(cutoff, errors="coerce")
    if isinstance(cutoff_ts, pd.Timestamp) and cutoff_ts.tz is not None:
        cutoff_ts = cutoff_ts.tz_convert(None)
    return cutoff_ts


def _rolling_mean(values: np.ndarray, window: int) -> float:
    if values.size == 0:
        return 0.0
    take = min(window, values.size)
    return float(values[-take:].mean()) if take else 0.0


def compute_history_features(
    *,
    stat_type: str,
    line_score: float,
    cutoff: datetime | None,
    player_logs: pd.DataFrame,
    league_mean: float,
    windows: tuple[int, int, int] = (3, 5, 10),
    min_std: float = 1e-6,
) -> dict[str, float]:
    diff = stat_diff_components(stat_type)
    components = stat_components(stat_type)
    if diff is None and not components:
        return {
            "hist_n": 0.0,
            "hist_mean": 0.0,
            "hist_std": 1.0,
            "league_mean": float(league_mean),
            "mu_stab": float(league_mean),
            "p_hist_over": 0.5,
            "z_line": 0.0,
            "rest_days": 0.0,
            "is_back_to_back": 0.0,
            "stat_mean_3": 0.0,
            "stat_mean_5": 0.0,
            "stat_mean_10": 0.0,
            "minutes_mean_3": 0.0,
            "minutes_mean_5": 0.0,
            "minutes_mean_10": 0.0,
        }

    cutoff_ts = _cutoff_date(cutoff)
    hist = player_logs
    if cutoff_ts is not None:
        hist = hist[hist["game_date"] < cutoff_ts]

    n = len(hist)
    if n == 0:
        hist_mean = 0.0
        hist_std = 1.0
        wins = 0
        rest_days = 0.0
        is_back_to_back = 0.0
        vals = np.zeros((0,), dtype=np.float32)
        mins = np.zeros((0,), dtype=np.float32)
    else:
        if diff is not None:
            base_col, sub_col = diff
            if base_col not in hist.columns or sub_col not in hist.columns:
                return {
                    "hist_n": 0.0,
                    "hist_mean": 0.0,
                    "hist_std": 1.0,
                    "league_mean": float(league_mean),
                    "mu_stab": float(league_mean),
                    "p_hist_over": 0.5,
                    "z_line": 0.0,
                    "rest_days": 0.0,
                    "is_back_to_back": 0.0,
                    "stat_mean_3": 0.0,
                    "stat_mean_5": 0.0,
                    "stat_mean_10": 0.0,
                    "minutes_mean_3": 0.0,
                    "minutes_mean_5": 0.0,
                    "minutes_mean_10": 0.0,
                }
            vals = (hist[base_col].fillna(0) - hist[sub_col].fillna(0)).to_numpy(dtype=np.float32)
        else:
            assert components is not None
            missing = [col for col in components if col not in hist.columns]
            if missing:
                return {
                    "hist_n": 0.0,
                    "hist_mean": 0.0,
                    "hist_std": 1.0,
                    "league_mean": float(league_mean),
                    "mu_stab": float(league_mean),
                    "p_hist_over": 0.5,
                    "z_line": 0.0,
                    "rest_days": 0.0,
                    "is_back_to_back": 0.0,
                    "stat_mean_3": 0.0,
                    "stat_mean_5": 0.0,
                    "stat_mean_10": 0.0,
                    "minutes_mean_3": 0.0,
                    "minutes_mean_5": 0.0,
                    "minutes_mean_10": 0.0,
                }
            vals = hist[components].fillna(0).sum(axis=1).to_numpy(dtype=np.float32)

        mins = hist["minutes"].fillna(0).to_numpy(dtype=np.float32)
        hist_mean = float(vals.mean())
        hist_std = float(vals.std(ddof=0)) if n > 1 else 1.0
        wins = int((vals > line_score).sum())
        rest_days = 0.0
        is_back_to_back = 0.0
        last_game_date = hist["game_date"].iloc[-1]
        if cutoff_ts is not None and pd.notna(last_game_date):
            diff_days = (cutoff_ts.date() - last_game_date.date()).days
            rest_days = max(diff_days - 1, 0)
            is_back_to_back = 1.0 if rest_days == 0 else 0.0

    stabilization = float(STABILIZATION_GAMES.get(stat_type, 10.0))
    mu_stab = (
        (n * hist_mean + stabilization * league_mean) / (n + stabilization)
        if n + stabilization > 0
        else league_mean
    )
    p_hist_over = (wins + 1.0) / (n + 2.0)
    z_line = (line_score - mu_stab) / (hist_std + min_std)

    stat_mean_3 = _rolling_mean(vals, windows[0])
    stat_mean_5 = _rolling_mean(vals, windows[1])
    stat_mean_10 = _rolling_mean(vals, windows[2])
    minutes_mean_3 = _rolling_mean(mins, windows[0])
    minutes_mean_5 = _rolling_mean(mins, windows[1])
    minutes_mean_10 = _rolling_mean(mins, windows[2])

    return {
        "hist_n": float(n),
        "hist_mean": float(hist_mean),
        "hist_std": float(hist_std),
        "league_mean": float(league_mean),
        "mu_stab": float(mu_stab),
        "p_hist_over": float(p_hist_over),
        "z_line": float(z_line),
        "rest_days": float(rest_days),
        "is_back_to_back": float(is_back_to_back),
        "stat_mean_3": float(stat_mean_3),
        "stat_mean_5": float(stat_mean_5),
        "stat_mean_10": float(stat_mean_10),
        "minutes_mean_3": float(minutes_mean_3),
        "minutes_mean_5": float(minutes_mean_5),
        "minutes_mean_10": float(minutes_mean_10),
    }
