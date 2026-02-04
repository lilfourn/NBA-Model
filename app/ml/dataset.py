from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.utils.names import normalize_name
from app.ml.feature_engineering import compute_history_features, compute_league_means, prepare_gamelogs
from app.ml.stat_mappings import STAT_COLUMNS, normalize_stat_type, stat_value_from_row

COMBO_SPLIT = re.compile(r"\s*\+\s*")


def compute_actual_value(row: pd.Series) -> float | None:
    return stat_value_from_row(row.get("stat_type"), row)


def _load_team_abbrev_overrides() -> dict[str, str]:
    path = Path(settings.team_abbrev_overrides_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return {str(k).upper(): str(v).upper() for k, v in data.items() if k and v}
    if isinstance(data, list):
        overrides: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or item.get("source_abbr")
            target = item.get("normalized") or item.get("normalized_abbr") or item.get("target")
            if source and target:
                overrides[str(source).upper()] = str(target).upper()
        return overrides
    return {}

def _load_name_overrides() -> dict[str, str]:
    path = Path(settings.player_name_overrides_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        overrides: dict[str, str] = {}
        for raw_name, normalized in data.items():
            key = normalize_name(raw_name)
            value = normalize_name(normalized)
            if key and value:
                overrides[key] = value
        return overrides
    return {}


def _build_team_abbrev_cte(overrides: dict[str, str]) -> tuple[str, dict[str, Any]]:
    if not overrides:
        return (
            "team_abbrev_map as (select null::text as source_abbr, null::text as normalized_abbr where false)",
            {},
        )

    params: dict[str, Any] = {}
    values_sql: list[str] = []
    for idx, (source, normalized) in enumerate(sorted(overrides.items())):
        src_key = f"src_{idx}"
        dst_key = f"dst_{idx}"
        params[src_key] = source
        params[dst_key] = normalized
        values_sql.append(f"(:{src_key}, :{dst_key})")

    cte = "team_abbrev_map as (select * from (values " + ", ".join(values_sql) + ") as m(source_abbr, normalized_abbr))"
    return cte, params


def load_training_data(engine: Engine) -> pd.DataFrame:
    overrides = _load_team_abbrev_overrides()
    cte, params = _build_team_abbrev_cte(overrides)
    name_overrides = _load_name_overrides()
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
            pl.name as player_name,
            pl.combo as is_combo,
            pl.name_key as prizepicks_name_key,
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
    base_df = pd.read_sql(query, engine, params=params)
    if base_df.empty:
        return base_df

    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players"),
        engine,
    )
    if nba_players.empty:
        return base_df

    def normalize_component(name: str) -> str | None:
        key = normalize_name(name)
        if not key:
            return None
        return name_overrides.get(key, key)

    component_rows: list[dict[str, Any]] = []
    for row in base_df.itertuples(index=False):
        raw_name = row.player_name or ""
        is_combo = bool(row.is_combo) or ("+" in raw_name)
        parts = [raw_name] if not is_combo else [p for p in COMBO_SPLIT.split(raw_name) if p.strip()]
        if not parts:
            continue
        for part in parts:
            component_rows.append(
                {
                    "snapshot_id": row.snapshot_id,
                    "projection_id": row.projection_id,
                    "nba_game_id": row.nba_game_id,
                    "component_name_key": normalize_component(part),
                }
            )

    components_df = pd.DataFrame(component_rows)
    if components_df.empty:
        return base_df

    components_df = components_df.dropna(subset=["component_name_key", "nba_game_id"])
    if components_df.empty:
        return base_df

    expected_counts = components_df.groupby(["snapshot_id", "projection_id"])["component_name_key"].count()

    merged = components_df.merge(
        nba_players,
        left_on="component_name_key",
        right_on="name_key",
        how="left",
    )
    merged = merged.dropna(subset=["nba_player_id"])
    if merged.empty:
        return base_df

    game_ids = sorted(merged["nba_game_id"].unique().tolist())
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
        return base_df

    merged = merged.merge(
        stats,
        on=["nba_player_id", "nba_game_id"],
        how="left",
    )

    stat_cols = STAT_COLUMNS
    merged["has_stats"] = merged[stat_cols].notna().any(axis=1)
    merged = merged[merged["has_stats"]]
    if merged.empty:
        return base_df

    matched_counts = merged.groupby(["snapshot_id", "projection_id"])["nba_player_id"].count()
    expected_aligned = expected_counts.reindex(matched_counts.index)
    valid_ids = matched_counts[matched_counts == expected_aligned].index
    if valid_ids.empty:
        return base_df

    aggregated = (
        merged.set_index(["snapshot_id", "projection_id"])
        .loc[valid_ids]
        .reset_index()
        .groupby(["snapshot_id", "projection_id"], as_index=False)[stat_cols]
        .sum(min_count=1)
    )

    full_df = base_df.merge(aggregated, on=["snapshot_id", "projection_id"], how="inner")
    full_df = _add_history_features(full_df, engine)
    return full_df


def _map_nba_player_ids(frame: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    nba_players = pd.read_sql(
        text("select id as nba_player_id, name_key from nba_players"),
        engine,
    )
    if nba_players.empty:
        frame["nba_player_id"] = None
        return frame

    name_overrides = _load_name_overrides()

    def normalize_component(name: str) -> str | None:
        key = normalize_name(name)
        if not key:
            return None
        return name_overrides.get(key, key)
    frame = frame.copy()
    frame["normalized_name_key"] = frame["player_name"].apply(normalize_component)
    merged = frame.merge(
        nba_players,
        left_on="normalized_name_key",
        right_on="name_key",
        how="left",
    )
    return merged


def _load_gamelogs(engine: Engine) -> pd.DataFrame:
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


def _add_history_features(frame: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = frame.copy()
    frame = _map_nba_player_ids(frame, engine)
    gamelogs = prepare_gamelogs(_load_gamelogs(engine))
    league_means = compute_league_means(gamelogs)
    logs_by_player = {
        str(player_id): group.reset_index(drop=True)
        for player_id, group in gamelogs.groupby("player_id", sort=False)
    }

    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce")
    start_time = None
    if "start_time" in frame.columns:
        start_time = pd.to_datetime(frame["start_time"], errors="coerce")
    cutoff_times = []
    for idx in range(len(frame)):
        fetch = fetched_at.iloc[idx] if fetched_at is not None else None
        start = start_time.iloc[idx] if start_time is not None else None
        if pd.notna(fetch) and pd.notna(start):
            cutoff_times.append(min(fetch, start))
        else:
            cutoff_times.append(fetch if pd.notna(fetch) else start)
    frame["cutoff_time"] = cutoff_times

    extras_rows: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        nba_player_id = getattr(row, "nba_player_id", None)
        if nba_player_id is None:
            extras_rows.append({})
            continue
        stat_type = getattr(row, "stat_type", None)
        stat_key = normalize_stat_type(stat_type)
        line_score = getattr(row, "line_score", None)
        cutoff = getattr(row, "cutoff_time", None)
        logs = logs_by_player.get(str(nba_player_id), pd.DataFrame(columns=gamelogs.columns))
        extras_rows.append(
            compute_history_features(
                stat_type=str(stat_type) if stat_type is not None else "",
                line_score=float(line_score) if line_score is not None else 0.0,
                cutoff=cutoff,
                player_logs=logs,
                league_mean=float(league_means.get(str(stat_key or ""), 0.0)),
            )
        )

    extras_df = pd.DataFrame(extras_rows)
    for key in extras_df.columns:
        frame[key] = extras_df[key]

    frame = frame.replace([float("inf"), float("-inf")], pd.NA)
    # Ensure feature columns exist even when we couldn't compute for a row.
    for col in [
        "hist_n",
        "hist_mean",
        "hist_std",
        "league_mean",
        "mu_stab",
        "p_hist_over",
        "z_line",
        "rest_days",
        "is_back_to_back",
        "stat_mean_3",
        "stat_mean_5",
        "stat_mean_10",
        "minutes_mean_3",
        "minutes_mean_5",
        "minutes_mean_10",
    ]:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame
