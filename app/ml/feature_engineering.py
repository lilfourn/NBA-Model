from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.modeling.stabilization import STABILIZATION_GAMES
from app.ml.stat_mappings import (
    SPECIAL_DIFFS,
    STAT_COMPONENTS,
    WEIGHTED_SUMS,
    stat_components,
    stat_diff_components,
    stat_weighted_components,
)


@dataclass
class _GamelogCacheEntry:
    frame: pd.DataFrame
    checked_at: float


_GAMELOG_FRAME_CACHE: dict[str, _GamelogCacheEntry] = {}
_CACHE_CHECK_TTL_SECONDS = 300.0


def clear_gamelog_frame_cache(database_url: str | None = None) -> None:
    if database_url is None:
        _GAMELOG_FRAME_CACHE.clear()
        return
    _GAMELOG_FRAME_CACHE.pop(str(database_url), None)


def prepare_gamelogs(frame: pd.DataFrame) -> pd.DataFrame:
    logs = frame.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    # Ensure numeric columns are numeric to avoid pandas downcasting warnings and
    # to keep feature math stable.
    for col in logs.columns:
        if col in {"player_id", "game_date"}:
            continue
        logs[col] = pd.to_numeric(logs[col], errors="coerce")
    return logs.sort_values(["player_id", "game_date"])


def load_gamelogs_frame(engine: Engine) -> pd.DataFrame:
    cache_key = str(engine.url)
    now = monotonic()
    cached = _GAMELOG_FRAME_CACHE.get(cache_key)
    if cached is not None and (now - cached.checked_at) < _CACHE_CHECK_TTL_SECONDS:
        return cached.frame

    query = text(
        """
        select
            s.player_id,
            ng.game_date as game_date,
            s.minutes,
            s.points,
            s.rebounds,
            s.assists,
            s.steals,
            s.blocks,
            s.turnovers,
            s.fg3m,
            s.fg3a,
            s.fgm,
            s.fga,
            s.ftm,
            s.fta,
            cast(s.stats_json->>'OREB' as float) as oreb,
            cast(s.stats_json->>'DREB' as float) as dreb,
            cast(s.stats_json->>'PF' as float) as pf,
            cast(s.stats_json->>'DUNKS' as float) as dunks
        from nba_player_game_stats s
        join nba_games ng on ng.id = s.game_id
        """
    )
    frame = pd.read_sql(query, engine)
    _GAMELOG_FRAME_CACHE[cache_key] = _GamelogCacheEntry(
        frame=frame,
        checked_at=now,
    )
    return frame


def build_logs_by_player(
    gamelogs: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    logs_by_player: dict[str, pd.DataFrame] = {}
    log_dates_by_player: dict[str, np.ndarray] = {}
    for player_id, group in gamelogs.groupby("player_id", sort=False):
        logs = group.reset_index(drop=True)
        key = str(player_id)
        logs_by_player[key] = logs
        log_dates_by_player[key] = logs["game_date"].to_numpy(dtype="datetime64[ns]")
    return logs_by_player, log_dates_by_player


def slice_player_logs_before_cutoff(
    player_logs: pd.DataFrame,
    log_dates: np.ndarray,
    cutoff: datetime | None,
) -> pd.DataFrame:
    if player_logs.empty:
        return player_logs
    cutoff_ts = _cutoff_date(cutoff)
    if cutoff_ts is None or pd.isna(cutoff_ts):
        return player_logs
    cutoff_np = np.datetime64(cutoff_ts.to_datetime64())
    end = int(np.searchsorted(log_dates, cutoff_np, side="left"))
    if end <= 0:
        return player_logs.iloc[0:0]
    if end >= len(player_logs):
        return player_logs
    return player_logs.iloc[:end]


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

    for stat_key, weights in WEIGHTED_SUMS.items():
        if not weights:
            continue
        missing = [col for col in weights.keys() if col not in gamelogs.columns]
        if missing:
            continue
        series = 0.0
        for col, weight in weights.items():
            series = series + (gamelogs[col].fillna(0) * float(weight))
        means[stat_key] = float(series.mean()) if len(gamelogs) else 0.0
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


def _empty_history(league_mean: float) -> dict[str, float]:
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
        "trend_slope": 0.0,
        "stat_cv": 0.0,
        "recent_vs_season": 1.0,
        "minutes_trend": 1.0,
        "stat_std_5": 0.0,
    }


def compute_history_features(
    *,
    stat_type: str,
    line_score: float,
    cutoff: datetime | None,
    player_logs: pd.DataFrame,
    league_mean: float,
    windows: tuple[int, int, int] = (3, 5, 10),
    min_std: float = 1e-6,
    logs_prefiltered: bool = False,
) -> dict[str, float]:
    diff = stat_diff_components(stat_type)
    components = stat_components(stat_type)
    weights = stat_weighted_components(stat_type)
    if diff is None and not components and not weights:
        return _empty_history(league_mean)

    cutoff_ts = _cutoff_date(cutoff)
    hist = player_logs
    if cutoff_ts is not None and not logs_prefiltered:
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
                return _empty_history(league_mean)
            vals = (hist[base_col].fillna(0) - hist[sub_col].fillna(0)).to_numpy(dtype=np.float32)
        elif weights:
            missing = [col for col in weights.keys() if col not in hist.columns]
            if missing:
                return _empty_history(league_mean)
            series = 0.0
            for col, weight in weights.items():
                series = series + (hist[col].fillna(0) * float(weight))
            vals = series.to_numpy(dtype=np.float32)
        else:
            assert components is not None
            missing = [col for col in components if col not in hist.columns]
            if missing:
                return _empty_history(league_mean)
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
    z_line = max(-10.0, min(10.0, z_line))

    stat_mean_3 = _rolling_mean(vals, windows[0])
    stat_mean_5 = _rolling_mean(vals, windows[1])
    stat_mean_10 = _rolling_mean(vals, windows[2])
    minutes_mean_3 = _rolling_mean(mins, windows[0])
    minutes_mean_5 = _rolling_mean(mins, windows[1])
    minutes_mean_10 = _rolling_mean(mins, windows[2])

    # --- Phase 2: Advanced temporal features ---

    # Trend slope: linear regression slope over last 10 games (positive = improving).
    trend_slope = 0.0
    if vals.size >= 3:
        recent = vals[-min(10, vals.size):]
        x = np.arange(len(recent), dtype=np.float32)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom > 0:
            trend_slope = float(((x - x_mean) * (recent - recent.mean())).sum() / denom)

    # Coefficient of variation (consistency): lower = more consistent player.
    stat_cv = 0.0
    if vals.size >= 3 and hist_mean > 0:
        stat_cv = float(hist_std / abs(hist_mean))

    # Recent vs season ratio: stat_mean_3 / hist_mean. >1 = hot streak, <1 = cold.
    recent_vs_season = 1.0
    if hist_mean > 0 and stat_mean_3 > 0:
        recent_vs_season = float(stat_mean_3 / hist_mean)

    # Minutes trend: recent 3-game avg vs 10-game avg ratio.
    minutes_trend = 1.0
    if minutes_mean_10 > 0 and minutes_mean_3 > 0:
        minutes_trend = float(minutes_mean_3 / minutes_mean_10)

    # Stat variance in last 5 games (raw, not normalized).
    stat_std_5 = 0.0
    if vals.size >= 2:
        recent5 = vals[-min(5, vals.size):]
        stat_std_5 = float(recent5.std(ddof=0))

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
        "trend_slope": float(trend_slope),
        "stat_cv": float(stat_cv),
        "recent_vs_season": float(recent_vs_season),
        "minutes_trend": float(minutes_trend),
        "stat_std_5": float(stat_std_5),
    }
