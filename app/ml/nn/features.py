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
    compute_actual_value,
)
from app.ml.feature_engineering import (
    compute_history_features,
    compute_league_means,
    prepare_gamelogs,
)
from app.ml.stat_mappings import normalize_stat_type, stat_components, stat_diff_components
from app.utils.names import normalize_name


@dataclass
class TrainingData:
    frame: pd.DataFrame
    numeric: pd.DataFrame
    sequences: np.ndarray
    cat_maps: dict[str, dict[str, int]]
    numeric_cols: list[str]


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


def _player_logs_map(gamelogs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(player_id): group.reset_index(drop=True)
        for player_id, group in gamelogs.groupby("player_id", sort=False)
    }


def build_history_features_for_row(
    row: pd.Series,
    player_logs: pd.DataFrame,
    league_means: dict[str, float],
    *,
    L: int = 10,
    min_std: float = 1e-6,
) -> tuple[dict[str, float], np.ndarray]:
    stat_type = str(row.get("stat_type") or "")
    line_score = _safe_numeric(row.get("line_score"), 0.0)
    cutoff = row.get("cutoff_time")
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
    )

    hist = player_logs
    if cutoff is not None:
        hist = hist[hist["game_date"] < cutoff]
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
        where lower(coalesce(p.attributes->>'odds_type', 'standard')) = 'standard'
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
            cast(s.stats_json->>'PF' as float) as pf
        from nba_player_game_stats s
        join nba_games ng on ng.id = s.game_id
        """
    )
    return pd.read_sql(query, engine)


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
    frame["actual_value"] = frame.apply(compute_actual_value, axis=1)
    frame = frame.dropna(subset=["line_score", "actual_value"])
    frame["over"] = (frame["actual_value"] > frame["line_score"]).astype(int)

    # Match the pregame inference regime.
    frame = frame[frame["minutes_to_start"].fillna(0) >= 0]
    frame = frame[frame["is_live"].fillna(False) == False]  # noqa: E712
    frame = frame[frame["in_game"].fillna(False) == False]  # noqa: E712

    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce")
    start_time = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["cutoff_time"] = [
        _cutoff_time(fetch, start)
        for fetch, start in zip(fetched_at, start_time)
    ]

    gamelogs = prepare_gamelogs(load_gamelogs(engine))
    league_means = compute_league_means(gamelogs)
    logs_by_player = _player_logs_map(gamelogs)

    extras_rows: list[dict[str, float]] = []
    sequences: list[np.ndarray] = []
    for _, row in frame.iterrows():
        player_id = row.get("nba_player_id")
        logs = logs_by_player.get(str(player_id), pd.DataFrame(columns=gamelogs.columns))
        extras, seq = build_history_features_for_row(
            row,
            logs,
            league_means,
            L=history_len,
        )
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
          and lower(coalesce(p.attributes->>'odds_type', 'standard')) = 'standard'
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
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    frame = build_inference_frame(engine, snapshot_id)
    if frame.empty:
        return frame, pd.DataFrame(), np.zeros((0, history_len, 2), dtype=np.float32)

    frame = frame.copy()
    name_overrides = _load_name_overrides()
    frame = _apply_name_overrides(frame, name_overrides)
    frame = _map_nba_player_ids(frame, engine)
    frame = frame.dropna(subset=["nba_player_id"])
    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce")
    start_time = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["cutoff_time"] = [
        _cutoff_time(fetch, start)
        for fetch, start in zip(fetched_at, start_time)
    ]

    gamelogs = prepare_gamelogs(load_gamelogs(engine))
    league_means = compute_league_means(gamelogs)
    logs_by_player = _player_logs_map(gamelogs)

    extras_rows: list[dict[str, float]] = []
    sequences: list[np.ndarray] = []
    for _, row in frame.iterrows():
        player_id = row.get("nba_player_id")
        logs = logs_by_player.get(str(player_id), pd.DataFrame(columns=gamelogs.columns))
        extras, seq = build_history_features_for_row(
            row,
            logs,
            league_means,
            L=history_len,
        )
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

    for cat_key, mapping in cat_maps.items():
        if cat_key not in frame.columns:
            frame[cat_key] = None
        frame[cat_key] = frame[cat_key].astype(str)

    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)
    return frame.reset_index(drop=True), numeric.reset_index(drop=True), sequences_arr
