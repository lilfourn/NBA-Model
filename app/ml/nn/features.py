from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.dataset import (
    _build_team_abbrev_cte,
    _load_name_overrides,
    _load_team_abbrev_overrides,
)
from app.ml.feature_engineering import (
    build_logs_by_player,
    compute_history_features,
    compute_league_means,
    load_gamelogs_frame,
    prepare_gamelogs,
    slice_player_logs_before_cutoff,
)
from app.ml.opponent_features import (
    build_opponent_defensive_averages,
    build_opponent_defensive_ranks,
    compute_opponent_features,
    _load_team_game_stats,
)
from app.ml.stat_mappings import (
    normalize_stat_type,
    stat_components,
    stat_diff_components,
    stat_value_from_row,
)
from app.utils.names import normalize_name


@dataclass
class TrainingData:
    frame: pd.DataFrame
    numeric: pd.DataFrame
    sequences: np.ndarray
    cat_maps: dict[str, dict[str, int]]
    numeric_cols: list[str]
    numeric_stats: dict[str, tuple[float, float]] | None = None


# Columns that are binary/categorical flags and should NOT be standardized.
_SKIP_STANDARDIZE = {
    "is_promo", "is_live", "in_game", "today", "odds_type",
    "is_back_to_back", "is_home",
}


def _compute_numeric_stats(numeric: pd.DataFrame) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in numeric.columns:
        if col in _SKIP_STANDARDIZE:
            continue
        mean = float(numeric[col].mean())
        std = float(numeric[col].std())
        if std < 1e-8:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def _apply_numeric_stats(numeric: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    numeric = numeric.copy()
    for col, (mean, std) in stats.items():
        if col in numeric.columns:
            numeric[col] = (numeric[col] - mean) / std
    return numeric


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_cat_map(values: Iterable[Any]) -> dict[str, int]:
    unique = sorted({str(value) for value in values if value is not None})
    return {value: idx + 1 for idx, value in enumerate(unique)}


def _cutoff_time(fetched_at: datetime | None, start_time: datetime | None) -> datetime | None:
    if fetched_at and start_time:
        return min(fetched_at, start_time)
    return fetched_at or start_time


def build_history_features_for_row(
    row: Any,
    player_logs: pd.DataFrame,
    league_means: dict[str, float],
    *,
    L: int = 10,
    min_std: float = 1e-6,
    logs_prefiltered: bool = False,
) -> tuple[dict[str, float], np.ndarray]:
    stat_type = str(getattr(row, "stat_type", "") or "")
    line_score = _safe_numeric(getattr(row, "line_score", None), 0.0)
    cutoff = getattr(row, "cutoff_time", None)
    if cutoff is not None and not isinstance(cutoff, pd.Timestamp):
        cutoff = pd.to_datetime(cutoff, errors="coerce")
    if isinstance(cutoff, pd.Timestamp) and cutoff.tz is not None:
        cutoff = cutoff.tz_convert(None)

    stat_key = normalize_stat_type(stat_type)
    stats = compute_history_features(
        stat_type=stat_type,
        line_score=line_score,
        cutoff=cutoff,
        player_logs=player_logs,
        league_mean=float(league_means.get(str(stat_key or ""), 0.0)),
        logs_prefiltered=logs_prefiltered,
    )

    hist = player_logs
    if cutoff is not None and not logs_prefiltered:
        # Normalize to start of day to exclude same-day games (leakage fix)
        cutoff_day = cutoff.normalize() if hasattr(cutoff, 'normalize') else cutoff
        hist = hist[hist["game_date"] < cutoff_day]
    if len(hist) == 0:
        seq = np.zeros((L, 2), dtype=np.float32)
    else:
        diff = stat_diff_components(stat_type)
        components = stat_components(stat_type)
        if diff is None and not components:
            seq = np.zeros((L, 2), dtype=np.float32)
        else:
            if diff is not None:
                base_col, sub_col = diff
                if base_col not in hist.columns or sub_col not in hist.columns:
                    seq = np.zeros((L, 2), dtype=np.float32)
                else:
                    vals = (hist[base_col].fillna(0) - hist[sub_col].fillna(0)).to_numpy(dtype=np.float32)
                    mins = hist["minutes"].fillna(0).to_numpy(dtype=np.float32)
                    take = min(L, len(hist))
                    seq = np.zeros((L, 2), dtype=np.float32)
                    seq[-take:, 0] = vals[-take:]
                    seq[-take:, 1] = mins[-take:]
            else:
                assert components is not None
                if any(col not in hist.columns for col in components):
                    seq = np.zeros((L, 2), dtype=np.float32)
                else:
                    vals = hist[components].fillna(0).sum(axis=1).to_numpy(dtype=np.float32)
                    mins = hist["minutes"].fillna(0).to_numpy(dtype=np.float32)
                    take = min(L, len(hist))
                    seq = np.zeros((L, 2), dtype=np.float32)
                    seq[-take:, 0] = vals[-take:]
                    seq[-take:, 1] = mins[-take:]

    return stats, seq


def load_nn_training_frame(engine: Engine) -> pd.DataFrame:
    overrides = _load_team_abbrev_overrides()
    cte, params = _build_team_abbrev_cte(overrides)
    query = text(
        f"""
        with {cte}
        select
            pf.snapshot_id,
            pf.projection_id,
            pf.player_id,
            pf.game_id,
            pf.line_score,
            pf.line_score_prev,
            pf.line_score_delta,
            pf.line_movement,
            pf.stat_type,
            pf.projection_type,
            pf.odds_type,
            pf.trending_count,
            pf.is_promo,
            pf.is_live,
            pf.in_game,
            pf.today,
            pf.minutes_to_start,
            pf.fetched_at,
            pf.start_time,
            pl.name_key as prizepicks_name_key,
            pl.display_name as player_name,
            pl.combo as combo,
            ng.id as nba_game_id
        from projection_features pf
        join projections p
            on p.snapshot_id = pf.snapshot_id
            and p.projection_id = pf.projection_id
        join players pl on pl.id = pf.player_id
        join games g on g.id = pf.game_id
        left join teams ht on ht.id = g.home_team_id
        left join teams at on at.id = g.away_team_id
        left join team_abbrev_map htm on htm.source_abbr = upper(ht.abbreviation)
        left join team_abbrev_map atm on atm.source_abbr = upper(at.abbreviation)
        join nba_games ng
            on ng.game_date = (g.start_time at time zone 'America/New_York')::date
            and ng.home_team_abbreviation = upper(coalesce(htm.normalized_abbr, ht.abbreviation))
            and ng.away_team_abbreviation = upper(coalesce(atm.normalized_abbr, at.abbreviation))
        where coalesce(p.odds_type, 0) = 0
          and lower(coalesce(p.event_type, p.attributes->>'event_type', '')) <> 'combo'
        """
    )
    return pd.read_sql(query, engine, params=params)


def _apply_name_overrides(frame: pd.DataFrame, name_overrides: dict[str, str]) -> pd.DataFrame:
    def normalize_key(value: Any) -> str | None:
        key = normalize_name(str(value)) if value is not None else None
        if not key:
            return None
        return name_overrides.get(key, key)

    frame = frame.copy()
    frame["normalized_name_key"] = frame["prizepicks_name_key"].apply(normalize_key)
    return frame


def _map_nba_player_ids(frame: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players"),
        engine,
    )
    if nba_players.empty:
        frame["nba_player_id"] = None
        return frame
    merged = frame.merge(
        nba_players,
        left_on="normalized_name_key",
        right_on="name_key",
        how="left",
    )
    return merged


def load_gamelogs(engine: Engine) -> pd.DataFrame:
    return load_gamelogs_frame(engine)


def build_training_data(
    *,
    engine: Engine,
    history_len: int = 10,
) -> TrainingData:
    frame = load_nn_training_frame(engine)
    if frame.empty:
        return TrainingData(
            frame=frame,
            numeric=pd.DataFrame(),
            sequences=np.zeros((0, history_len, 2), dtype=np.float32),
            cat_maps={},
            numeric_cols=[],
        )

    frame = frame.copy()
    frame = frame[frame["combo"].isna() | (frame["combo"] == False)]
    name_overrides = _load_name_overrides()
    frame = _apply_name_overrides(frame, name_overrides)
    frame = _map_nba_player_ids(frame, engine)
    frame = frame.dropna(subset=["nba_player_id", "nba_game_id"])

    if frame.empty:
        return TrainingData(
            frame=frame,
            numeric=pd.DataFrame(),
            sequences=np.zeros((0, history_len, 2), dtype=np.float32),
            cat_maps={},
            numeric_cols=[],
        )

    game_ids = sorted(frame["nba_game_id"].dropna().unique().tolist())
    stats = pd.read_sql(
        text(
            """
            select
                game_id as nba_game_id,
                player_id as nba_player_id,
                points,
                rebounds,
                assists,
                steals,
                blocks,
                turnovers,
                fg3m,
                fg3a,
                fgm,
                fga,
                ftm,
                fta,
                cast(stats_json->>'OREB' as float) as oreb,
                cast(stats_json->>'DREB' as float) as dreb,
                cast(stats_json->>'PF' as float) as pf
            from nba_player_game_stats
            where game_id = any(:game_ids)
            """
        ),
        engine,
        params={"game_ids": game_ids},
    )
    if stats.empty:
        return TrainingData(
            frame=pd.DataFrame(),
            numeric=pd.DataFrame(),
            sequences=np.zeros((0, history_len, 2), dtype=np.float32),
            cat_maps={},
            numeric_cols=[],
        )

    frame = frame.merge(stats, on=["nba_player_id", "nba_game_id"], how="left")
    frame["actual_value"] = [
        stat_value_from_row(getattr(row, "stat_type", None), row)
        for row in frame.itertuples(index=False)
    ]
    frame = frame.dropna(subset=["line_score", "actual_value"])
    frame["over"] = (frame["actual_value"] > frame["line_score"]).astype(int)

    # Match the pregame inference regime.
    frame = frame[frame["minutes_to_start"].fillna(0) >= 0]
    frame = frame[frame["is_live"].fillna(False) == False]  # noqa: E712
    frame = frame[frame["in_game"].fillna(False) == False]  # noqa: E712

    # Deduplicate: keep earliest snapshot per player+game+stat to prevent
    # the same prediction leaking across train/test via multiple snapshots.
    dedup_cols = ["nba_player_id", "nba_game_id", "stat_type"]
    if all(c in frame.columns for c in dedup_cols):
        frame = frame.sort_values("fetched_at").drop_duplicates(subset=dedup_cols, keep="first")

    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce")
    start_time = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["cutoff_time"] = [
        _cutoff_time(fetch, start)
        for fetch, start in zip(fetched_at, start_time)
    ]

    gamelogs = prepare_gamelogs(load_gamelogs(engine))
    league_means = compute_league_means(gamelogs)
    logs_by_player, log_dates_by_player = build_logs_by_player(gamelogs)
    empty_logs = gamelogs.iloc[0:0]
    empty_dates = np.array([], dtype="datetime64[ns]")

    # Opponent defensive context
    team_game_stats = _load_team_game_stats(engine)
    opp_def_avgs = build_opponent_defensive_averages(team_game_stats)
    opp_def_ranks = build_opponent_defensive_ranks(opp_def_avgs)

    nba_player_teams = pd.read_sql(
        text("select id as nba_player_id, team_abbreviation from nba_players"),
        engine,
    )
    player_team_map: dict[str, str] = {}
    for _r in nba_player_teams.itertuples(index=False):
        if _r.team_abbreviation:
            player_team_map[str(_r.nba_player_id)] = str(_r.team_abbreviation).upper()

    game_teams: dict[str, tuple[str, str]] = {}
    for _r in team_game_stats.drop_duplicates("game_id").itertuples(index=False):
        game_teams[str(_r.game_id)] = (
            str(_r.home_team_abbreviation).upper(),
            str(_r.away_team_abbreviation).upper(),
        )

    extras_rows: list[dict[str, float]] = []
    sequences: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        player_key = str(getattr(row, "nba_player_id", ""))
        cutoff = getattr(row, "cutoff_time", None)
        logs = logs_by_player.get(player_key, empty_logs)
        log_dates = log_dates_by_player.get(player_key, empty_dates)
        logs = slice_player_logs_before_cutoff(logs, log_dates, cutoff)
        extras, seq = build_history_features_for_row(
            row,
            logs,
            league_means,
            L=history_len,
            logs_prefiltered=True,
        )

        # Opponent features
        nba_game_id = getattr(row, "nba_game_id", None)
        player_team = player_team_map.get(player_key)
        opp_team = None
        is_home = None
        if nba_game_id and str(nba_game_id) in game_teams:
            home_abbr, away_abbr = game_teams[str(nba_game_id)]
            if player_team:
                if player_team == home_abbr:
                    opp_team = away_abbr
                    is_home = 1
                elif player_team == away_abbr:
                    opp_team = home_abbr
                    is_home = 0
        stat_type = str(getattr(row, "stat_type", "") or "")
        opp_feats = compute_opponent_features(
            player_team_abbr=player_team,
            opp_team_abbr=opp_team,
            game_date=cutoff,
            stat_type=stat_type,
            opp_def_avgs=opp_def_avgs,
            is_home=is_home,
            opp_def_ranks=opp_def_ranks,
        )
        extras.update(opp_feats)

        extras_rows.append(extras)
        sequences.append(seq)

    extras_df = pd.DataFrame(extras_rows)
    numeric_cols = [
        "line_score",
        "line_score_prev",
        "line_score_delta",
        "minutes_to_start",
        "odds_type",
        "trending_count",
        "is_promo",
        "is_live",
        "in_game",
        "today",
    ]
    numeric = pd.concat([frame[numeric_cols].reset_index(drop=True), extras_df], axis=1)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if "minutes_to_start" in numeric.columns:
        numeric["minutes_to_start"] = (
            numeric["minutes_to_start"].clip(lower=0.0, upper=360.0) / 360.0
        )
    if "trending_count" in numeric.columns:
        numeric["trending_count"] = np.log1p(
            numeric["trending_count"].clip(lower=0.0)
        )

    # Standardize continuous numeric features for NN stability.
    numeric_stats = _compute_numeric_stats(numeric)
    numeric = _apply_numeric_stats(numeric, numeric_stats)

    cat_maps = {
        "stat_type": _build_cat_map(frame["stat_type"]),
        "projection_type": _build_cat_map(frame["projection_type"]),
        "line_movement": _build_cat_map(frame["line_movement"]),
    }
    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)
    return TrainingData(
        frame=frame.reset_index(drop=True),
        numeric=numeric.reset_index(drop=True),
        sequences=sequences_arr,
        cat_maps=cat_maps,
        numeric_cols=list(numeric.columns),
        numeric_stats=numeric_stats,
    )


def build_inference_frame(engine: Engine, snapshot_id: str) -> pd.DataFrame:
    query = text(
        """
        select
            pf.snapshot_id,
            pf.projection_id,
            pf.player_id,
            pf.game_id,
            pf.line_score,
            pf.line_score_prev,
            pf.line_score_delta,
            pf.line_movement,
            pf.stat_type,
            pf.projection_type,
            pf.odds_type,
            pf.trending_count,
            pf.is_promo,
            pf.is_live,
            pf.in_game,
            pf.today,
            pf.minutes_to_start,
            pf.fetched_at,
            pf.start_time,
            pl.name_key as prizepicks_name_key,
            pl.display_name as player_name,
            pl.combo as combo
        from projection_features pf
        join projections p
            on p.snapshot_id = pf.snapshot_id
            and p.projection_id = pf.projection_id
        join players pl on pl.id = pf.player_id
        where pf.snapshot_id = :snapshot_id
          and coalesce(p.odds_type, 0) = 0
          and lower(coalesce(p.event_type, p.attributes->>'event_type', '')) <> 'combo'
          and (pl.combo is null or pl.combo = false)
          and (pf.is_live is null or pf.is_live = false)
          and (pf.in_game is null or pf.in_game = false)
          and (pf.minutes_to_start is null or pf.minutes_to_start >= 0)
        """
    )
    return pd.read_sql(query, engine, params={"snapshot_id": snapshot_id})


def build_inference_data(
    *,
    engine: Engine,
    snapshot_id: str,
    history_len: int,
    cat_maps: dict[str, dict[str, int]],
    numeric_stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    frame = build_inference_frame(engine, snapshot_id)
    if frame.empty:
        return frame, pd.DataFrame(), np.zeros((0, history_len, 2), dtype=np.float32)

    frame = frame.copy()
    name_overrides = _load_name_overrides()
    frame = _apply_name_overrides(frame, name_overrides)
    frame = _map_nba_player_ids(frame, engine)
    frame = frame.dropna(subset=["nba_player_id"]).copy()
    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce")
    start_time = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["cutoff_time"] = [
        _cutoff_time(fetch, start)
        for fetch, start in zip(fetched_at, start_time)
    ]

    gamelogs = prepare_gamelogs(load_gamelogs(engine))
    league_means = compute_league_means(gamelogs)
    logs_by_player, log_dates_by_player = build_logs_by_player(gamelogs)
    empty_logs = gamelogs.iloc[0:0]
    empty_dates = np.array([], dtype="datetime64[ns]")

    # Opponent defensive context for inference
    team_game_stats = _load_team_game_stats(engine)
    opp_def_avgs = build_opponent_defensive_averages(team_game_stats)
    opp_def_ranks_inf = build_opponent_defensive_ranks(opp_def_avgs)

    nba_player_teams_inf = pd.read_sql(
        text("select id as nba_player_id, team_abbreviation from nba_players"),
        engine,
    )
    player_team_map_inf: dict[str, str] = {}
    for _ri in nba_player_teams_inf.itertuples(index=False):
        if _ri.team_abbreviation:
            player_team_map_inf[str(_ri.nba_player_id)] = str(_ri.team_abbreviation).upper()

    game_teams_inf: dict[str, tuple[str, str]] = {}
    for _ri in team_game_stats.drop_duplicates("game_id").itertuples(index=False):
        game_teams_inf[str(_ri.game_id)] = (
            str(_ri.home_team_abbreviation).upper(),
            str(_ri.away_team_abbreviation).upper(),
        )

    # Fallback: PrizePicks game_id -> (home_abbr, away_abbr) for upcoming games
    pp_game_teams_inf: dict[str, tuple[str, str]] = {}
    if "game_id" in frame.columns:
        pp_gids = sorted({str(v) for v in frame["game_id"].dropna().unique()})
        if pp_gids:
            overrides_inf = _load_team_abbrev_overrides()
            override_map_inf = {k.upper(): v.upper() for k, v in overrides_inf.items()}
            pp_games_inf = pd.read_sql(
                text(
                    """
                    select g.id as game_id, ht.abbreviation as home_abbr, at.abbreviation as away_abbr
                    from games g
                    left join teams ht on ht.id = g.home_team_id
                    left join teams at on at.id = g.away_team_id
                    where g.id = any(:game_ids)
                    """
                ),
                engine,
                params={"game_ids": pp_gids},
            )
            for _ri in pp_games_inf.itertuples(index=False):
                h = override_map_inf.get(str(_ri.home_abbr or "").upper(), str(_ri.home_abbr or "").upper())
                a = override_map_inf.get(str(_ri.away_abbr or "").upper(), str(_ri.away_abbr or "").upper())
                if h and a:
                    pp_game_teams_inf[str(_ri.game_id)] = (h, a)

    extras_rows: list[dict[str, float]] = []
    sequences: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        player_key = str(getattr(row, "nba_player_id", ""))
        cutoff = getattr(row, "cutoff_time", None)
        logs = logs_by_player.get(player_key, empty_logs)
        log_dates = log_dates_by_player.get(player_key, empty_dates)
        logs = slice_player_logs_before_cutoff(logs, log_dates, cutoff)
        extras, seq = build_history_features_for_row(
            row,
            logs,
            league_means,
            L=history_len,
            logs_prefiltered=True,
        )

        # Opponent features for inference
        nba_game_id_inf = getattr(row, "nba_game_id", None)
        pp_game_id_inf = getattr(row, "game_id", None)
        player_team_inf = player_team_map_inf.get(player_key)
        opp_team_inf = None
        is_home_inf = None
        resolved_inf = False
        if nba_game_id_inf and str(nba_game_id_inf) in game_teams_inf:
            home_a, away_a = game_teams_inf[str(nba_game_id_inf)]
            if player_team_inf:
                if player_team_inf == home_a:
                    opp_team_inf = away_a
                    is_home_inf = 1
                    resolved_inf = True
                elif player_team_inf == away_a:
                    opp_team_inf = home_a
                    is_home_inf = 0
                    resolved_inf = True
        if not resolved_inf and pp_game_id_inf and str(pp_game_id_inf) in pp_game_teams_inf:
            home_a, away_a = pp_game_teams_inf[str(pp_game_id_inf)]
            if player_team_inf:
                if player_team_inf == home_a:
                    opp_team_inf = away_a
                    is_home_inf = 1
                elif player_team_inf == away_a:
                    opp_team_inf = home_a
                    is_home_inf = 0
        stat_type_inf = str(getattr(row, "stat_type", "") or "")
        opp_feats_inf = compute_opponent_features(
            player_team_abbr=player_team_inf,
            opp_team_abbr=opp_team_inf,
            game_date=cutoff,
            stat_type=stat_type_inf,
            opp_def_avgs=opp_def_avgs,
            is_home=is_home_inf,
            opp_def_ranks=opp_def_ranks_inf,
        )
        extras.update(opp_feats_inf)

        extras_rows.append(extras)
        sequences.append(seq)

    extras_df = pd.DataFrame(extras_rows)
    numeric_cols = [
        "line_score",
        "line_score_prev",
        "line_score_delta",
        "minutes_to_start",
        "odds_type",
        "trending_count",
        "is_promo",
        "is_live",
        "in_game",
        "today",
    ]
    numeric = pd.concat([frame[numeric_cols].reset_index(drop=True), extras_df], axis=1)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if "minutes_to_start" in numeric.columns:
        numeric["minutes_to_start"] = (
            numeric["minutes_to_start"].clip(lower=0.0, upper=360.0) / 360.0
        )
    if "trending_count" in numeric.columns:
        numeric["trending_count"] = np.log1p(
            numeric["trending_count"].clip(lower=0.0)
        )

    # Apply same standardization as training.
    if numeric_stats:
        numeric = _apply_numeric_stats(numeric, numeric_stats)

    for cat_key, mapping in cat_maps.items():
        if cat_key not in frame.columns:
            frame[cat_key] = None
        frame[cat_key] = frame[cat_key].astype(str)

    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)
    return frame.reset_index(drop=True), numeric.reset_index(drop=True), sequences_arr
