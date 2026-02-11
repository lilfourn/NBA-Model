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

TEAM_GEO: dict[str, dict[str, float]] = {
    "ATL": {"lat": 33.76, "lon": -84.39, "tz": -5.0},
    "BOS": {"lat": 42.36, "lon": -71.06, "tz": -5.0},
    "BKN": {"lat": 40.68, "lon": -73.98, "tz": -5.0},
    "CHA": {"lat": 35.23, "lon": -80.84, "tz": -5.0},
    "CHI": {"lat": 41.88, "lon": -87.63, "tz": -6.0},
    "CLE": {"lat": 41.50, "lon": -81.69, "tz": -5.0},
    "DAL": {"lat": 32.78, "lon": -96.80, "tz": -6.0},
    "DEN": {"lat": 39.74, "lon": -104.99, "tz": -7.0},
    "DET": {"lat": 42.33, "lon": -83.05, "tz": -5.0},
    "GSW": {"lat": 37.77, "lon": -122.42, "tz": -8.0},
    "HOU": {"lat": 29.76, "lon": -95.37, "tz": -6.0},
    "IND": {"lat": 39.77, "lon": -86.16, "tz": -5.0},
    "LAC": {"lat": 34.05, "lon": -118.24, "tz": -8.0},
    "LAL": {"lat": 34.05, "lon": -118.24, "tz": -8.0},
    "MEM": {"lat": 35.15, "lon": -90.05, "tz": -6.0},
    "MIA": {"lat": 25.76, "lon": -80.19, "tz": -5.0},
    "MIL": {"lat": 43.04, "lon": -87.91, "tz": -6.0},
    "MIN": {"lat": 44.98, "lon": -93.27, "tz": -6.0},
    "NOP": {"lat": 29.95, "lon": -90.07, "tz": -6.0},
    "NYK": {"lat": 40.75, "lon": -73.99, "tz": -5.0},
    "OKC": {"lat": 35.47, "lon": -97.52, "tz": -6.0},
    "ORL": {"lat": 28.54, "lon": -81.38, "tz": -5.0},
    "PHI": {"lat": 39.95, "lon": -75.17, "tz": -5.0},
    "PHX": {"lat": 33.45, "lon": -112.07, "tz": -7.0},
    "POR": {"lat": 45.52, "lon": -122.68, "tz": -8.0},
    "SAC": {"lat": 38.58, "lon": -121.49, "tz": -8.0},
    "SAS": {"lat": 29.42, "lon": -98.49, "tz": -6.0},
    "TOR": {"lat": 43.65, "lon": -79.38, "tz": -5.0},
    "UTA": {"lat": 40.76, "lon": -111.89, "tz": -7.0},
    "WAS": {"lat": 38.91, "lon": -77.04, "tz": -5.0},
}


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
    # Normalize to start of day so same-day games are excluded.
    # game_date is midnight; without this, a cutoff of 14:00 on game day
    # would include the current game's stats → data leakage.
    cutoff_ts = cutoff_ts.normalize()
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


def build_league_means_timeline(gamelogs: pd.DataFrame) -> dict[str, Any]:
    """Build cumulative league means by game_date for leakage-safe lookups.

    Returns:
        {
            "dates": np.ndarray[datetime64[ns]],
            "sums": {stat_key: np.ndarray},
            "counts": {stat_key: np.ndarray},
            "global": {stat_key: float},
        }
    """
    logs = gamelogs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs = logs.dropna(subset=["game_date"]).sort_values("game_date")

    if logs.empty:
        return {
            "dates": np.array([], dtype="datetime64[ns]"),
            "sums": {},
            "counts": {},
            "global": {},
        }

    timeline_dates = np.sort(
        logs["game_date"].drop_duplicates().to_numpy(dtype="datetime64[ns]")
    )
    timeline_index = pd.to_datetime(timeline_dates)
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}

    def _register_stat(stat_key: str, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
        tmp = pd.DataFrame(
            {"game_date": logs["game_date"].values, "value": numeric.values}
        )
        daily = (
            tmp.groupby("game_date", sort=True, observed=False)["value"]
            .agg(["sum", "count"])
            .reindex(timeline_index, fill_value=0.0)
        )
        sums[stat_key] = daily["sum"].to_numpy(dtype=float).cumsum()
        counts[stat_key] = daily["count"].to_numpy(dtype=float).cumsum()

    for stat_key, cols in STAT_COMPONENTS.items():
        if not cols or any(col not in logs.columns for col in cols):
            continue
        _register_stat(stat_key, logs[cols].fillna(0).sum(axis=1))

    for stat_key, (base_col, sub_col) in SPECIAL_DIFFS.items():
        if base_col not in logs.columns or sub_col not in logs.columns:
            continue
        _register_stat(stat_key, logs[base_col].fillna(0) - logs[sub_col].fillna(0))

    for stat_key, weights in WEIGHTED_SUMS.items():
        if not weights:
            continue
        missing = [col for col in weights.keys() if col not in logs.columns]
        if missing:
            continue
        weighted = pd.Series(0.0, index=logs.index, dtype=float)
        for col, weight in weights.items():
            weighted = weighted + (logs[col].fillna(0) * float(weight))
        _register_stat(stat_key, weighted)

    return {
        "dates": timeline_dates,
        "sums": sums,
        "counts": counts,
        "global": compute_league_means(logs),
    }


def league_mean_before_cutoff(
    timeline: dict[str, Any],
    stat_key: str,
    cutoff: datetime | None,
    *,
    default: float = 0.0,
) -> float:
    """Lookup league mean strictly before cutoff date (same-day excluded)."""
    if not stat_key:
        return float(default)

    dates = timeline.get("dates")
    sums = timeline.get("sums", {})
    counts = timeline.get("counts", {})
    if dates is None or stat_key not in sums or stat_key not in counts:
        return float(default)

    cutoff_ts = _cutoff_date(cutoff)
    if cutoff_ts is None or pd.isna(cutoff_ts):
        global_means = timeline.get("global", {})
        return float(global_means.get(stat_key, default))

    cutoff_day = cutoff_ts.normalize()
    cutoff_np = np.datetime64(cutoff_day.to_datetime64())
    idx = int(np.searchsorted(dates, cutoff_np, side="left")) - 1
    if idx < 0:
        return float(default)

    denom = float(counts[stat_key][idx])
    if denom <= 0:
        return float(default)
    return float(sums[stat_key][idx] / denom)


def compute_player_usage(
    player_fga: float,
    player_fta: float,
    player_to: float,
    team_fga: float,
    team_fta: float,
    team_to: float,
) -> float:
    """Player usage rate = share of team possessions used.

    Usage = (player_FGA + 0.44*player_FTA + player_TO) / (team_FGA + 0.44*team_FTA + team_TO)
    Returns 0.0 if denominator is 0.
    """
    denom = team_fga + 0.44 * team_fta + team_to
    if denom == 0:
        return 0.0
    return float((player_fga + 0.44 * player_fta + player_to) / denom)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(radius_km * c)


def estimate_team_travel_context(
    team_game_stats: pd.DataFrame,
    *,
    team_abbreviation: str | None,
    cutoff: datetime | None,
    lookback_games: int = 7,
) -> dict[str, float]:
    """Estimate travel/fatigue proxies from recent team schedule transitions.

    Returns:
      - travel_km_7d: cumulative inter-game venue distance over recent games.
      - circadian_shift_hours: cumulative absolute timezone shift.
    """
    if (
        team_abbreviation is None
        or not str(team_abbreviation).strip()
        or team_game_stats.empty
    ):
        return {"travel_km_7d": 0.0, "circadian_shift_hours": 0.0}

    team = str(team_abbreviation).strip().upper()
    if team not in TEAM_GEO:
        return {"travel_km_7d": 0.0, "circadian_shift_hours": 0.0}

    cutoff_ts = _cutoff_date(cutoff)
    frame = team_game_stats[
        team_game_stats["team_abbreviation"].astype(str).str.upper() == team
    ].copy()
    if frame.empty:
        return {"travel_km_7d": 0.0, "circadian_shift_hours": 0.0}

    if cutoff_ts is not None:
        frame = frame[frame["game_date"] < cutoff_ts.normalize()]
    if frame.empty:
        return {"travel_km_7d": 0.0, "circadian_shift_hours": 0.0}

    frame = frame.sort_values("game_date").tail(max(2, int(lookback_games)))
    venues: list[str] = []
    for row in frame.itertuples(index=False):
        home_abbr = str(getattr(row, "home_team_abbreviation", "") or "").upper()
        away_abbr = str(getattr(row, "away_team_abbreviation", "") or "").upper()
        venue = home_abbr if home_abbr == team else away_abbr
        if venue and venue in TEAM_GEO:
            venues.append(venue)

    if len(venues) < 2:
        return {"travel_km_7d": 0.0, "circadian_shift_hours": 0.0}

    total_km = 0.0
    total_tz = 0.0
    prev = venues[0]
    for venue in venues[1:]:
        g1 = TEAM_GEO.get(prev)
        g2 = TEAM_GEO.get(venue)
        if g1 is not None and g2 is not None:
            total_km += _haversine_km(g1["lat"], g1["lon"], g2["lat"], g2["lon"])
            total_tz += abs(float(g2["tz"]) - float(g1["tz"]))
        prev = venue

    return {
        "travel_km_7d": round(float(total_km), 4),
        "circadian_shift_hours": round(float(total_tz), 4),
    }


def _cutoff_date(cutoff: datetime | None) -> datetime | None:
    if cutoff is None:
        return None
    cutoff_ts = (
        cutoff
        if isinstance(cutoff, pd.Timestamp)
        else pd.to_datetime(cutoff, errors="coerce")
    )
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
        "minutes_std_10": 0.0,
        "recent_load_3": 0.0,
        "role_stability": 0.0,
        "stat_std_5": 0.0,
        "stat_rate_per_min": 0.0,
        "line_vs_mean_ratio": 1.0,
        "hot_streak_count": 0.0,
        "cold_streak_count": 0.0,
        "season_game_number": 0.0,
        "forecast_edge": 0.0,
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
        cutoff_day = cutoff_ts.normalize()
        hist = hist[hist["game_date"] < cutoff_day]

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
            vals = (hist[base_col].fillna(0) - hist[sub_col].fillna(0)).to_numpy(
                dtype=np.float32
            )
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
        recent = vals[-min(10, vals.size) :]
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
    minutes_std_10 = 0.0
    if mins.size >= 2:
        minutes_std_10 = float(mins[-min(10, mins.size) :].std(ddof=0))
    recent_load_3 = float(mins[-min(3, mins.size) :].sum()) if mins.size > 0 else 0.0
    role_stability = float(1.0 / (1.0 + max(0.0, minutes_std_10)))

    # Stat variance in last 5 games (raw, not normalized).
    stat_std_5 = 0.0
    if vals.size >= 2:
        recent5 = vals[-min(5, vals.size) :]
        stat_std_5 = float(recent5.std(ddof=0))

    # Stat rate per minute (efficiency): recent 5-game stat / minutes ratio.
    stat_rate_per_min = 0.0
    if stat_mean_5 > 0 and minutes_mean_5 > 0:
        stat_rate_per_min = float(stat_mean_5 / minutes_mean_5)

    # Line vs historical mean ratio: how aggressive is the line relative to player average.
    line_vs_mean_ratio = 1.0
    if hist_mean > 0:
        line_vs_mean_ratio = float(line_score / hist_mean)

    # Line-agnostic momentum: consecutive recent games above/below player's own mean.
    # Uses the player's rolling average instead of the current line to avoid leakage.
    hot_streak_count = 0.0
    cold_streak_count = 0.0
    if vals.size > 0 and hist_mean > 0:
        for v in reversed(vals):
            if v > hist_mean:
                hot_streak_count += 1.0
            else:
                break
        for v in reversed(vals):
            if v <= hist_mean:
                cold_streak_count += 1.0
            else:
                break

    # Season game number: how many games this player has played this season (fatigue proxy).
    season_game_number = float(n)

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
        "minutes_std_10": float(minutes_std_10),
        "recent_load_3": float(recent_load_3),
        "role_stability": float(role_stability),
        "stat_std_5": float(stat_std_5),
        "stat_rate_per_min": float(stat_rate_per_min),
        "line_vs_mean_ratio": float(line_vs_mean_ratio),
        "hot_streak_count": float(hot_streak_count),
        "cold_streak_count": float(cold_streak_count),
        "season_game_number": float(season_game_number),
        "forecast_edge": float(mu_stab - line_score),
    }
