from __future__ import annotations

from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.stat_mappings import (
    STAT_COMPONENTS,
    SPECIAL_DIFFS,
    WEIGHTED_SUMS,
    normalize_stat_type,
)

_OPP_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_OPP_CACHE_TTL = 300.0

# Rolling window (games) for opponent defensive averages.
OPP_DEF_WINDOW = 15


def _load_team_game_stats(engine: Engine) -> pd.DataFrame:
    """Load per-team-per-game aggregated box scores with opponent info."""
    cache_key = str(engine.url)
    now = monotonic()
    cached = _OPP_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _OPP_CACHE_TTL:
        return cached[0]

    query = text(
        """
        select
            s.game_id,
            ng.game_date,
            s.team_abbreviation,
            ng.home_team_abbreviation,
            ng.away_team_abbreviation,
            sum(s.points)    as points,
            sum(s.rebounds)  as rebounds,
            sum(s.assists)   as assists,
            sum(s.steals)    as steals,
            sum(s.blocks)    as blocks,
            sum(s.turnovers) as turnovers,
            sum(s.fg3m)      as fg3m,
            sum(s.fg3a)      as fg3a,
            sum(s.fgm)       as fgm,
            sum(s.fga)       as fga,
            sum(s.ftm)       as ftm,
            sum(s.fta)       as fta,
            sum(cast(s.stats_json->>'OREB' as float)) as oreb,
            sum(cast(s.stats_json->>'DREB' as float)) as dreb,
            sum(cast(s.stats_json->>'PF' as float))   as pf,
            sum(cast(s.stats_json->>'DUNKS' as float)) as dunks
        from nba_player_game_stats s
        join nba_games ng on ng.id = s.game_id
        group by s.game_id, ng.game_date, s.team_abbreviation,
                 ng.home_team_abbreviation, ng.away_team_abbreviation
        """
    )
    frame = pd.read_sql(query, engine)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    for col in frame.columns:
        if col in {"game_id", "game_date", "team_abbreviation",
                    "home_team_abbreviation", "away_team_abbreviation"}:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    # Determine opponent abbreviation
    frame["opp_abbreviation"] = np.where(
        frame["team_abbreviation"] == frame["home_team_abbreviation"],
        frame["away_team_abbreviation"],
        frame["home_team_abbreviation"],
    )
    frame["is_home"] = (
        frame["team_abbreviation"] == frame["home_team_abbreviation"]
    ).astype(int)

    frame = frame.sort_values(["team_abbreviation", "game_date"])
    _OPP_CACHE[cache_key] = (frame, now)
    return frame


def _stat_value_series(frame: pd.DataFrame, stat_type: str) -> pd.Series | None:
    """Compute the stat value column for a given stat_type from a frame."""
    key = normalize_stat_type(stat_type)
    if not key:
        return None

    diff = SPECIAL_DIFFS.get(key)
    if diff is not None:
        base_col, sub_col = diff
        if base_col in frame.columns and sub_col in frame.columns:
            return frame[base_col].fillna(0) - frame[sub_col].fillna(0)
        return None

    weights = WEIGHTED_SUMS.get(key)
    if weights:
        series = pd.Series(0.0, index=frame.index)
        for col, w in weights.items():
            if col not in frame.columns:
                return None
            series = series + frame[col].fillna(0) * float(w)
        return series

    components = STAT_COMPONENTS.get(key)
    if components:
        missing = [c for c in components if c not in frame.columns]
        if missing:
            return None
        return frame[components].fillna(0).sum(axis=1)

    return None


def build_opponent_defensive_averages(
    team_game_stats: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    For each team, compute rolling defensive averages: how many of each stat
    opponents scored against them over the last OPP_DEF_WINDOW games.

    Returns dict: opp_abbreviation -> DataFrame with game_date and rolling means.
    """
    stat_cols = [
        "points", "rebounds", "assists", "steals", "blocks", "turnovers",
        "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
    ]

    # Group by opponent: "what did teams score AGAINST this opponent?"
    # If team A plays opponent B, team A's stats in that game are what B "allowed".
    result: dict[str, pd.DataFrame] = {}
    for opp_abbr, group in team_game_stats.groupby("opp_abbreviation", sort=False):
        sorted_g = group.sort_values("game_date")
        avail_cols = [c for c in stat_cols if c in sorted_g.columns]
        if not avail_cols:
            continue
        rolling = (
            sorted_g[avail_cols]
            .rolling(window=OPP_DEF_WINDOW, min_periods=3)
            .mean()
        )
        rolling.columns = [f"opp_def_{c}" for c in avail_cols]
        rolling["game_date"] = sorted_g["game_date"].values
        rolling["game_id"] = sorted_g["game_id"].values
        result[str(opp_abbr)] = rolling.reset_index(drop=True)

    return result


def build_opponent_defensive_ranks(
    opp_def_avgs: dict[str, pd.DataFrame],
) -> dict[str, dict[str, int]]:
    """Build per-stat ordinal rank (1=fewest allowed, 30=most allowed) for each team.

    Uses each team's latest available defensive average row.
    Returns: {stat_col: {team_abbr: rank}}
    """
    stat_cols = [
        "opp_def_points", "opp_def_rebounds", "opp_def_assists",
        "opp_def_steals", "opp_def_blocks", "opp_def_turnovers",
        "opp_def_fg3m", "opp_def_fgm",
    ]
    team_latest: dict[str, pd.Series] = {}
    for team_abbr, df in opp_def_avgs.items():
        if df.empty:
            continue
        team_latest[str(team_abbr)] = df.iloc[-1]

    ranks: dict[str, dict[str, int]] = {}
    for col in stat_cols:
        vals: dict[str, float] = {}
        for team_abbr, row in team_latest.items():
            if col in row.index and pd.notna(row[col]):
                vals[team_abbr] = float(row[col])
        if not vals:
            continue
        sorted_teams = sorted(vals.keys(), key=lambda t: vals[t])
        ranks[col] = {t: rank + 1 for rank, t in enumerate(sorted_teams)}
    return ranks


def compute_opponent_features(
    *,
    player_team_abbr: str | None,
    opp_team_abbr: str | None,
    game_date: Any,
    stat_type: str,
    opp_def_avgs: dict[str, pd.DataFrame],
    is_home: int | None,
    opp_def_ranks: dict[str, dict[str, int]] | None = None,
) -> dict[str, float]:
    """Compute opponent defensive features for a single row."""
    features: dict[str, float] = {
        "is_home": float(is_home) if is_home is not None else 0.5,
        "opp_def_stat_avg": 0.0,
        "opp_def_points_avg": 0.0,
        "opp_def_rebounds_avg": 0.0,
        "opp_def_assists_avg": 0.0,
        "opp_def_rank": 15.0,
    }

    if not opp_team_abbr or opp_team_abbr not in opp_def_avgs:
        return features

    opp_df = opp_def_avgs[opp_team_abbr]
    if opp_df.empty:
        return features

    # Find latest available defensive average before game_date
    if game_date is not None:
        game_date_ts = pd.to_datetime(game_date, errors="coerce")
        if pd.notna(game_date_ts):
            # Strip timezone to match tz-naive game_date column
            if hasattr(game_date_ts, "tz") and game_date_ts.tz is not None:
                game_date_ts = game_date_ts.tz_localize(None)
            # Normalize to start of day to exclude same-day games (leakage fix)
            game_date_ts = game_date_ts.normalize()
            mask = opp_df["game_date"] < game_date_ts
            opp_df = opp_df[mask]

    if opp_df.empty:
        return features

    latest = opp_df.iloc[-1]

    for base_col in ["points", "rebounds", "assists"]:
        col = f"opp_def_{base_col}"
        if col in latest.index:
            val = latest[col]
            if pd.notna(val):
                features[f"opp_def_{base_col}_avg"] = float(val)

    # Compute opponent defensive average for the specific stat type
    key = normalize_stat_type(stat_type)
    if key:
        components = STAT_COMPONENTS.get(key)
        if components:
            total = 0.0
            found = True
            for c in components:
                col = f"opp_def_{c}"
                if col in latest.index and pd.notna(latest[col]):
                    total += float(latest[col])
                else:
                    found = False
                    break
            if found:
                features["opp_def_stat_avg"] = total

        diff = SPECIAL_DIFFS.get(key)
        if diff is not None:
            base_col, sub_col = diff
            bc = f"opp_def_{base_col}"
            sc = f"opp_def_{sub_col}"
            if bc in latest.index and sc in latest.index:
                bv = latest[bc] if pd.notna(latest[bc]) else 0.0
                sv = latest[sc] if pd.notna(latest[sc]) else 0.0
                features["opp_def_stat_avg"] = float(bv) - float(sv)

        weights = WEIGHTED_SUMS.get(key)
        if weights:
            total = 0.0
            found = True
            for c, w in weights.items():
                col = f"opp_def_{c}"
                if col in latest.index and pd.notna(latest[col]):
                    total += float(w) * float(latest[col])
                else:
                    found = False
                    break
            if found:
                features["opp_def_stat_avg"] = total

    # Opponent defensive rank for the primary stat component (1=best D, 30=worst D).
    if opp_def_ranks and opp_team_abbr:
        key = normalize_stat_type(stat_type)
        if key:
            components = STAT_COMPONENTS.get(key)
            rank_col = f"opp_def_{components[0]}" if components else None
            if rank_col and rank_col in opp_def_ranks:
                features["opp_def_rank"] = float(
                    opp_def_ranks[rank_col].get(opp_team_abbr, 15)
                )

    return features


def load_game_context(engine: Engine) -> pd.DataFrame:
    """Load game-level context: home/away teams, for joining to projections."""
    query = text(
        """
        select
            ng.id as nba_game_id,
            ng.game_date,
            ng.home_team_abbreviation,
            ng.away_team_abbreviation
        from nba_games ng
        """
    )
    frame = pd.read_sql(query, engine)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    return frame
